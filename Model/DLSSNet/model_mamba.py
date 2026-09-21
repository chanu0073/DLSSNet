"""
Mamba-backboned variant of DLSSNet (Track C from the roadmap), following the
redesign sketched in `vit ap.docx`. Kept as a separate, additive model file
-- Model/DLSSNet/model_zhengjiao.py (the paper-matching architecture the
released checkpoints and notebooks depend on) is untouched.

What is Mamba here vs. what stays attention (and why):
  - FSSR (Embedding): Temporal conv -> Frequency-aware multi-branch conv ->
    Mamba block -> gated Dynamic State Refinement.            [Mamba]
  - TRE: Dynamic Task-Relevance Estimator (MLP + uncertainty). [new]
  - ABE / Contextual Embedding: Mamba block over the state sequence, in
    place of masked single-head self-attention.               [Mamba]
  - ABE / Subspace Encoding: the ORIGINAL attention readout, paper Eq. A.4
    (ClassificationTransHead, reused verbatim).               [attention]
  - ABE / Subspace Decoding: the ORIGINAL cross-attention (DecoderLayer,
    reused verbatim).                                          [attention]

The two subspace blocks are set-readout / expansion operations between the
T states and the m orthonormal basis vectors -- cross-attention-shaped, not
sequence-shaped. A Mamba subspace-encoding head (basis tokens appended to
the end of a causal scan) WAS built and measured first: the only route for
trial information to reach the tokens is the SSM state, whose cross-
position contribution scales like dt*B*C -- ~0.3% of the constant token/skip
terms at init with the small-dt initialization long memory requires. The
logits were input-independent to 4 decimals at init, and training either
crawled out or stuck at chance depending on seed/subject. The Eq. A.4
attention readout is a convex combination of the states, i.e. O(1) signal
at init. With 252 training trials per subject that difference is decisive.

Submodules are named and shaped to match model_zhengjiao.Net's interface
exactly (`.Embedding`, `.Encoder`, `.classhead`, `.Decoder`, same return
tuple shapes), so Trainer/trainer_corrected_protocol.py's CorrectedProtocolTrainer
works with this model unmodified -- this is a controlled comparison against
the same corrected-protocol baseline, same data split, same losses, only the
backbone differs.

Design decisions filling gaps the docx's diagrams leave unspecified:
  - Real S6 selective-scan Mamba block (Model/DLSSNet/mamba_layers.py),
    sequential (not the fused CUDA kernel) -- see that file's docstring.
  - Channel (spatial) collapse happens first, same as the original FSSR,
    then Temporal Conv -> Frequency-aware multi-branch conv -> Mamba ->
    gated refinement. The docx's diagram implies temporal-then-frequency
    conv before any spatial step; collapsing channels first is a
    simplification that avoids running multi-branch frequency convs across
    an extra spatial dimension for no clear benefit.
  - Subspace Decoding Block stays cross-attention (DecoderLayer, unchanged,
    imported from layers_zhengjiao). Reconstructing many timepoints from a
    handful of compressed basis vectors is a cross-attention-shaped problem
    (different Q/K-V lengths) that Mamba has no standard mechanism for; the
    docx's own diagram doesn't resolve this either, just labels the block
    "Mamba State Space Decoder" without specifying the expansion mechanism.
  - Contextual Embedding's task-relevance mask (Eq. A.3 in the paper) has no
    direct analogue for a causal scan the way it does for attention (there's
    no attention matrix to bias). The scan therefore sees the soft Eq. A.1
    relevance-scaled sequence; the hard mask is kept only for the decoder's
    reconstruction target, as in the original. (See MambaEncoderLayer for
    why zeroing states before the scan -- the first attempt -- is a trap.)
"""
import torch
import torch.nn as nn

from Model.DLSSNet.mamba_layers import MambaBlock
from Model.DLSSNet.layers_zhengjiao import (
    ClassificationTransHead,
    Conv2dWithConstraint,
    DecoderLayer,
    Initializer,
    PoswiseFeedForwardNet,
    p_filter,
)
from Model.DLSSNet.model_zhengjiao import PositionalEncoding


class DynamicTaskRelevanceEstimator(nn.Module):
    """Replaces Pool's fixed 2-way linear relevance projection with an MLP,
    plus an uncertainty head that down-weights the relevance scaling for
    states the model isn't confident about -- the docx's "MLP + Uncertainty-
    Aware Gating." Same return signature as the original Pool, so it's a
    drop-in replacement."""

    def __init__(self, in_dim, hidden, low_p, dropout):
        super().__init__()
        self.low_p = low_p
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.uncertainty_head = nn.Linear(in_dim, 1)

        # Same init the original Pool uses (glorot weights, ZERO biases).
        # This matters: the relevance decision `p_ir > low_p` is a hard
        # threshold, and with PyTorch's default nonzero bias init the final
        # layer's bias (~+-0.18) dwarfs the input-dependent part of the
        # score (~0.02), so at init either ~0% or ~95% of ALL states got
        # flagged irrelevant depending purely on the sign of a random bias
        # -- a per-seed coin flip (observed: seed 2 masked 95% of states
        # across every subject). Zero bias makes the decision input-driven
        # and roughly balanced from the start, as in the original.
        Initializer.weights_init(self.mlp)
        Initializer.weights_init(self.uncertainty_head)
        with torch.no_grad():
            # Start confident (uncertainty ~0.12) so (1 - uncertainty) doesn't
            # halve the signal at init on top of the sigmoid relevance gate.
            self.uncertainty_head.bias.fill_(-2.0)

    def forward(self, h):
        z = self.drop(h)
        raw_scores = self.mlp(z)  # (B,T,2): [score_tr, score_ir]
        scores = self.softmax(raw_scores)
        uncertainty = self.sigmoid(self.uncertainty_head(z)).squeeze(-1)  # (B,T) in [0,1]
        scaler = self.sigmoid(raw_scores[:, :, 0]) * (1.0 - uncertainty)
        mask, mask_index = p_filter(scores=scores, low_p=self.low_p)
        z = torch.mul(z, torch.unsqueeze(scaler, -1))
        return z, mask, mask_index, scaler, scores


class FrequencyAwareConv(nn.Module):
    """Multi-branch depthwise temporal conv with several kernel sizes
    (different receptive fields ~ different frequency sensitivities,
    filter-bank style, echoing FBCNet/EEGNet's multi-band idea already
    present elsewhere in this repo's baselines), concatenated and mixed back
    to d_model via a pointwise conv."""

    def __init__(self, d_model, kernel_sizes=(8, 16, 32)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ZeroPad2d((k - 1, 0, 0, 0)),
                    Conv2dWithConstraint(
                        d_model, d_model, kernel_size=(1, k), bias=True, p=2, max_norm=0.25
                    ),
                    nn.ELU(),
                )
                for k in kernel_sizes
            ]
        )
        self.mix = nn.Conv2d(d_model * len(kernel_sizes), d_model, kernel_size=(1, 1), bias=True)
        # BatchNorm after the last constrained conv, mirroring where the
        # original FSSR normalizes. Without it the 0.25-norm-constrained
        # branches left the embedding at std ~0.1, and the positional
        # encoding (std ~0.7) added downstream swamped the trial-dependent
        # signal 6:1 (measured across-batch/overall std ratio fell from 0.42
        # to 0.07 at the +posenc step).
        self.norm = nn.Sequential(nn.BatchNorm2d(d_model), nn.ELU())

    def forward(self, x):
        # x: (B, d_model, 1, T)
        branch_outs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outs, dim=1)
        return self.norm(self.mix(x))


class DynamicStateRefinement(nn.Module):
    """GRU-cell-style gated residual refinement applied to the Mamba
    backbone's output before it becomes the Functional State Sequence."""

    def __init__(self, d_model):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model)
        self.refine_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        refine = torch.tanh(self.refine_proj(x))
        return x + gate * refine


class MambaPatchEmbedding(nn.Module):
    """FSSR module, modified: spatial collapse (unchanged from the original)
    -> Temporal Conv -> Frequency-aware multi-branch conv -> Mamba block ->
    Dynamic State Refinement -> Functional State Sequence."""

    def __init__(self, len_window, num_channel, d_model, freq_kernel_sizes, ssm_state_dim, mamba_expand, mamba_conv_kernel, dropout):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            # No time-axis padding here: kernel width is 1 in time (this conv
            # only collapses the channel/spatial axis), so any pad added here
            # would pass straight through unconsumed. The temporal_conv below
            # pads and consumes it in one self-contained step instead.
            Conv2dWithConstraint(1, d_model, kernel_size=(num_channel, 1), bias=True, p=1, doWeightNorm=True, max_norm=1),
            nn.ELU(),
        )
        self.temporal_conv = nn.Sequential(
            nn.ZeroPad2d((len_window - 1, 0, 0, 0)),
            Conv2dWithConstraint(d_model, d_model, kernel_size=(1, len_window), bias=True, p=2, max_norm=0.25),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.Dropout(0.25),  # the original FSSR hardcodes 0.25 here regardless of config.dropout; match it
        )
        self.freq_conv = FrequencyAwareConv(d_model, freq_kernel_sizes)
        self.mamba = MambaBlock(d_model, d_state=ssm_state_dim, expand=mamba_expand, d_conv=mamba_conv_kernel)
        self.refine = DynamicStateRefinement(d_model)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # (B,1,C,T)
        x = self.spatial_conv(x)  # (B,d_model,1,T)
        x = self.temporal_conv(x)  # (B,d_model,1,T)
        x = self.freq_conv(x)  # (B,d_model,1,T)
        x = torch.squeeze(x, 2)  # (B,d_model,T)
        x = torch.transpose(x, 1, 2)  # (B,T,d_model)
        x = self.mamba(x)
        x = self.refine(x)
        return x


class MambaEncoderLayer(nn.Module):
    """Contextual Embedding Block, modified: TRE's fixed relevance scoring ->
    DynamicTaskRelevanceEstimator; masked self-attention -> Mamba block.

    The scan sees the soft-scaled sequence (the paper's Eq. A.1 relevance
    gate, differentiable, never exactly zero). The hard Eq. A.3 mask is NOT
    applied to the scan input -- it is returned only for the decoder's
    reconstruction target (`y = embeded_x * mask_index` in Net.forward),
    exactly as the original Encoder does. Zeroing states before the scan
    was tried first and is a trap: the threshold is non-differentiable, so
    once most states are flagged irrelevant the encoder receives near-zero
    input AND no gradient can ever un-flag them -- an absorbing state. The
    original attention mask never had this problem because it only shapes
    which state pairs interact; it never removes values from the stream."""

    def __init__(self, d_model, ssm_state_dim, mamba_expand, mamba_conv_kernel, tre_hidden, low_p, dropout, d_ff):
        super().__init__()
        self.tre = DynamicTaskRelevanceEstimator(d_model, tre_hidden, low_p, dropout)
        self.mamba = MambaBlock(d_model, d_state=ssm_state_dim, expand=mamba_expand, d_conv=mamba_conv_kernel)
        self.pos_ffn = PoswiseFeedForwardNet(d_model=d_model, d_ff=d_ff)

    def forward(self, x):
        z, mask, mask_index, scaler, scores = self.tre(x)
        pooled_x = z.clone()
        x = self.mamba(z)
        x = self.pos_ffn(x)
        return x, mask, mask_index, pooled_x, scaler, scores


class MambaEncoder(nn.Module):
    def __init__(self, num_layers, d_model, ssm_state_dim, mamba_expand, mamba_conv_kernel, tre_hidden, low_p, dropout, d_ffrate, max_len):
        super().__init__()
        d_ff = d_ffrate * d_model
        self.posenc = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                MambaEncoderLayer(d_model, ssm_state_dim, mamba_expand, mamba_conv_kernel, tre_hidden, low_p, dropout, d_ff)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.posenc(x)
        mask = mask_index = pooled_x = scaler = scores = None
        for layer in self.layers:
            x, mask, mask_index, pooled_x, scaler, scores = layer(x)
        return x, mask, mask_index, pooled_x, scaler, scores


class Net(nn.Module):
    def __init__(
        self,
        num_channels,
        len_window,
        d_model,
        num_frame,
        encoder_num_layers,
        low_p,
        dropout,
        transformerparwiseforward_dimrat,
        statenum,
        num_class,
        ssm_state_dim=16,
        mamba_expand=2,
        mamba_conv_kernel=4,
        freq_kernel_sizes=(8, 16, 32),
        tre_hidden=32,
        num_head=1,  # unused; kept for call-signature compatibility with Net(**MODEL_KW)
        frame_stride=1,  # unused for the same reason -- Mamba backbone doesn't downsample
    ) -> None:
        super().__init__()
        self.Embedding = MambaPatchEmbedding(
            len_window=len_window,
            num_channel=num_channels,
            d_model=d_model,
            freq_kernel_sizes=freq_kernel_sizes,
            ssm_state_dim=ssm_state_dim,
            mamba_expand=mamba_expand,
            mamba_conv_kernel=mamba_conv_kernel,
            dropout=dropout,
        )
        self.Encoder = MambaEncoder(
            num_layers=encoder_num_layers,
            d_model=d_model,
            ssm_state_dim=ssm_state_dim,
            mamba_expand=mamba_expand,
            mamba_conv_kernel=mamba_conv_kernel,
            tre_hidden=tre_hidden,
            low_p=low_p,
            dropout=dropout,
            d_ffrate=transformerparwiseforward_dimrat,
            max_len=num_frame,
        )
        # Subspace Encoding: the ORIGINAL attention readout (paper Eq. A.4),
        # reused verbatim -- see the module docstring for why a Mamba head
        # was tried and dropped.
        self.classhead = ClassificationTransHead(
            d_model=d_model,
            d_k=d_model,
            num_heads=num_head,
            dropout=dropout,
            state_classes=statenum,
            n_classes=num_class,
        )
        d_ff = transformerparwiseforward_dimrat * d_model
        self.Decoder = DecoderLayer(d_model=d_model, d_k=d_model, n_heads=1, d_ff=d_ff, dropout=dropout)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.Embedding(x)
        embeded_x = torch.detach(x)
        x, attn, mask_index, pooled_x, scaler, scores = self.Encoder(x)
        y = torch.mul(embeded_x, mask_index)
        out, basis, attn_class, states = self.classhead(x)
        dec_x, dec_attn = self.Decoder(y, basis)
        all_att = {"encoder_att": attn, "state_attn": attn_class, "decoder_attn": dec_attn}
        return all_att, out, dec_x, embeded_x, basis, pooled_x, states, scaler, scores

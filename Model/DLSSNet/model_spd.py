"""
DLSSNet + covariance (SPD) readout branch.

Motivation (MAtt, Pan et al., NeurIPS 2022, Table 2): on BCI IV-2a, the SAME
2-conv feature extractor scores 49.2% with ordinary self-attention on top
and 74.7% with manifold attention -- the difference is representing feature
segments as covariance (SPD) matrices, i.e. second-order spatial statistics,
the classic CSP-style motor-imagery feature. DLSSNet's state sequence is
first-order (per-timepoint amplitudes) and has to learn covariance
structure implicitly. This variant adds an explicit covariance readout.

What changes vs. the original (everything else is reused verbatim):
  - Nothing in FSSR / TRE / Contextual Embedding / Subspace Encoding /
    Decoding. The functional state sequence, the masks, the reconstruction
    loss, the orthogonality loss and therefore the whole interpretability
    pipeline are untouched.
  - The classifier's input is [subspace states (as before) ; SPD features],
    where the SPD features come from the TRE-scaled state sequence Y'
    (the same "task-highly-relevant" sequence the paper's own analysis
    protocol uses): linear projection k -> d_u, split into m temporal
    segments, per-segment covariance (MAtt's exact E2R recipe: center,
    SCM, trace-normalize, +eps*I), Log-Cholesky vectorization, LayerNorm.

Log-Cholesky (Lin, SIAM J. Matrix Anal. Appl. 2019) instead of MAtt's
Log-Euclidean (eig-based) map: it is a valid Riemannian geometry on SPD
matrices, and its backward pass is numerically stable, whereas eigh's
gradient explodes on near-repeated eigenvalues. The unconstrained linear
projection before the covariance is a BiMap (W C W^T) without the Stiefel
constraint, which lets this train with plain Adam rather than MAtt's
Riemannian MixOptimizer.

Interface: `forward_for_loss(x)` returns (out, dec_x, y, basis) for
Trainer/trainer_corrected_protocol.py; `forward(x)` keeps the original
9-tuple for the notebooks.
"""
import torch
import torch.nn as nn

from Model.DLSSNet.layers_zhengjiao import PatchEmbedding, DecoderLayer, ClassificationTransHead
from Model.DLSSNet.model_zhengjiao import Encoder


class SPDReadout(nn.Module):
    def __init__(self, in_dim, spd_dim=20, n_segments=3, eps=1e-4):
        super().__init__()
        self.proj = nn.Linear(in_dim, spd_dim, bias=False)
        self.n_segments = n_segments
        self.eps = eps
        self.spd_dim = spd_dim
        self.out_dim = n_segments * spd_dim * (spd_dim + 1) // 2
        self.norm = nn.LayerNorm(self.out_dim)
        tril = torch.tril_indices(spd_dim, spd_dim, offset=-1)
        self.register_buffer("tril_rows", tril[0])
        self.register_buffer("tril_cols", tril[1])

    def _covariance(self, z):
        # z: (B, S, L, d) -> (B, S, d, d), MAtt's signal2spd recipe, all
        # segments in one batched op chain (the per-segment Python loop was
        # pure CPU dispatch overhead: ~100 tiny kernels per step).
        z = z - z.mean(dim=2, keepdim=True)
        cov = z.transpose(-1, -2) @ z / (z.shape[2] - 1)
        tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        cov = cov / tr * self.spd_dim  # trace-normalize; scaled so eigenvalues average 1
        eye = torch.eye(self.spd_dim, device=z.device, dtype=z.dtype)
        return cov + self.eps * eye

    def _log_cholesky(self, cov):
        # Log-Cholesky map: L = chol(C); vector = [log diag(L), strict-lower(L)].
        # cholesky_ex, not cholesky: the latter's error check forces a
        # CPU<->GPU sync per call, which stalls the pipeline and made epochs
        # ~3.5x slower even though the factorization itself costs <0.1 ms.
        # With +eps*I on a trace-normalized SCM the factorization cannot fail.
        L, _ = torch.linalg.cholesky_ex(cov)
        log_diag = torch.log(L.diagonal(dim1=-2, dim2=-1))
        lower = L[..., self.tril_rows, self.tril_cols]
        return torch.cat([log_diag, lower], dim=-1)

    def forward(self, seq):
        # seq: (B, T, in_dim) -> (B, out_dim)
        z = self.proj(seq)
        B, T, d = z.shape
        S = self.n_segments
        L = T // S  # equal-length segments; at most S-1 trailing samples dropped (438 = 3 x 146 exactly)
        z = z[:, : S * L].reshape(B, S, L, d)
        feats = self._log_cholesky(self._covariance(z))  # (B, S, d(d+1)/2), one batched factorization
        return self.norm(feats.reshape(B, -1))


class SPDClassificationHead(ClassificationTransHead):
    """Original subspace-encoding attention head; the FC classifier now takes
    [states ; SPD features]."""

    def __init__(self, d_model, d_k, num_heads, dropout, state_classes, n_classes, spd_dim_out):
        super().__init__(d_model, d_k, num_heads, dropout, state_classes, n_classes)
        self.fc = nn.Sequential(
            nn.Linear(state_classes * d_model + spd_dim_out, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x, spd_feat):
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x, summary_token], dim=1)
        x, attn = self.classATTN(x, x, x)
        indx = -summary_token.shape[1]
        states = x[:, indx:, :].contiguous()
        out = self.fc(torch.cat([states.view(x.size(0), -1), spd_feat], dim=-1))
        return out, summary_token, attn, states


class Net(nn.Module):
    def __init__(
        self,
        num_channels,
        len_window,
        d_model,
        frame_stride,
        num_frame,
        num_head,
        encoder_num_layers,
        low_p,
        dropout,
        transformerparwiseforward_dimrat,
        statenum,
        num_class,
        spd_dim=20,
        spd_segments=3,
        spd_eps=1e-4,
    ) -> None:
        super().__init__()
        self.Embedding = PatchEmbedding(
            len_window=len_window, num_channel=num_channels, d_model=d_model, frame_stride=frame_stride, dropout=dropout
        )
        self.Encoder = Encoder(
            num_layers=encoder_num_layers,
            d_model=d_model,
            d_k=d_model,
            num_heads=num_head,
            dropout=dropout,
            d_ffrate=transformerparwiseforward_dimrat,
            max_len=num_frame,
            low_p=low_p,
        )
        self.spd = SPDReadout(d_model, spd_dim=spd_dim, n_segments=spd_segments, eps=spd_eps)
        self.classhead = SPDClassificationHead(
            d_model=d_model,
            d_k=d_model,
            num_heads=num_head,
            dropout=dropout,
            state_classes=statenum,
            n_classes=num_class,
            spd_dim_out=self.spd.out_dim,
        )
        d_ff = transformerparwiseforward_dimrat * d_model
        self.Decoder = DecoderLayer(d_model=d_model, d_k=d_model, n_heads=num_head, d_ff=d_ff, dropout=dropout)

    def _run(self, x):
        x = x.to(torch.float32)
        x = self.Embedding(x)
        embeded_x = torch.detach(x)
        x, attn, mask_index, pooled_x, scaler, scores = self.Encoder(x)
        y = torch.mul(embeded_x, mask_index)
        spd_feat = self.spd(pooled_x)  # covariance readout of the TRE-scaled sequence Y'
        out, basis, attn_class, states = self.classhead(x, spd_feat)
        dec_x, dec_attn = self.Decoder(y, basis)
        return dict(
            out=out, dec_x=dec_x, y=y, basis=basis, embeded_x=embeded_x, pooled_x=pooled_x,
            states=states, scaler=scaler, scores=scores, attn=attn, attn_class=attn_class, dec_attn=dec_attn,
        )

    def forward_for_loss(self, x):
        r = self._run(x)
        return r["out"], r["dec_x"], r["y"], r["basis"]

    def forward(self, x):
        r = self._run(x)
        all_att = {"encoder_att": r["attn"], "state_attn": r["attn_class"], "decoder_attn": r["dec_attn"]}
        return all_att, r["out"], r["dec_x"], r["embeded_x"], r["basis"], r["pooled_x"], r["states"], r["scaler"], r["scores"]

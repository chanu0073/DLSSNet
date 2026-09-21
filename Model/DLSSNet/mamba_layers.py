"""
A real selective state-space block (Mamba / S6, Gu & Dao 2023).

This is the piece the DLSSNet paper cites (S4/HiPPO/Mamba, Section 2.2.1) but
doesn't actually implement, falling back to small-kernel convolutions "due to
training difficulty." Rather than pulling in `mamba-ssm`'s CUDA/Triton build
(another version-conflict risk on top of the scipy/numpy issues already hit
on this machine), the recurrence is computed with a pure-PyTorch parallel
(Hillis-Steele) scan: a naive Python loop over T=438 timesteps turned out to
be far too slow in practice (single-epoch times blew up to minutes, driven
by per-step GPU kernel-launch overhead rather than actual compute) -- the
parallel scan reduces the sequential dependency from O(T) to O(log2 T) (~9
steps for T=438), each step fully vectorized over the time dimension.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _parallel_linear_scan(a, b):
    """Hillis-Steele inclusive parallel scan for the affine recurrence
    h_t = a_t * h_{t-1} + b_t (h_{-1} = 0), applied along dim=1 (time).

    The combination of two consecutive operators (a_earlier, b_earlier) then
    (a_later, b_later) is (a_later*a_earlier, a_later*b_earlier + b_later) --
    associative, so an inclusive prefix scan over these pairs yields, after
    log2(T) doubling steps, b_t = h_t directly at every position.

    a, b: (B, T, D, N). Returns h: (B, T, D, N) with h[:, t] = h_t.
    """
    T = a.shape[1]
    s = 1
    while s < T:
        a_shift = F.pad(a[:, :-s], (0, 0, 0, 0, s, 0), value=1.0)  # identity=1 for the product
        b_shift = F.pad(b[:, :-s], (0, 0, 0, 0, s, 0), value=0.0)  # identity=0 for the sum
        b = a * b_shift + b
        a = a * a_shift
        s *= 2
    return b


class SelectiveSSM(nn.Module):
    """Core S6 recurrence: per-channel diagonal state, with input-dependent
    step size (Delta), input matrix (B) and output matrix (C) -- the
    "selective" part of Mamba, as opposed to a plain time-invariant S4.

    h_t = exp(Delta_t * A) * h_{t-1} + (Delta_t * B_t) * x_t
    y_t = C_t . h_t + D * x_t
    """

    def __init__(self, d_inner, d_state=16, dt_rank=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        dt_rank = dt_rank or max(1, d_inner // 16)

        # A is kept strictly negative (via -exp(A_log)) for a stable
        # recurrence, HiPPO-style init (larger negative for higher state
        # index -> faster-decaying modes, slower ones capture longer range).
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Low-rank projection for Delta (standard Mamba parameter-efficiency
        # trick), direct projections for B and C.
        self.dt_proj_in = nn.Linear(d_inner, dt_rank, bias=False)
        self.dt_proj_out = nn.Linear(dt_rank, d_inner, bias=True)
        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)

        # Critical init (standard Mamba practice, Gu & Dao 2023 reference
        # implementation): with default nn.Linear bias~=0, softplus(0)~=0.69
        # is a large per-step decay -- compounded over T=438 sequential
        # steps, deltaA=exp(delta*A) underflows to exactly 0 within the
        # first few dozen steps, so the scan forgets everything before that
        # almost immediately. Verified empirically: this caused the
        # classification head (whose summary tokens sit at the END of a
        # 461-step causal sequence) to be unable to learn at all -- stuck
        # exactly at chance accuracy, confirmed via an 8-sample overfit test
        # where cls_loss never moved while dec_loss/reg_loss (which don't
        # depend on this scan) trained normally.
        #
        # Initializing the bias so delta starts small (dt_min..dt_max,
        # log-uniform) keeps deltaA close to 1 (near-identity dynamics) at
        # the start of training, so the network can actually learn
        # appropriate per-channel decay rates via gradient descent instead
        # of starting from a state where long-range signal is already gone.
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(d_inner) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))) + torch.log(torch.tensor(dt_min)))
        dt = dt.clamp(min=dt_init_floor)
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^{-1}(dt), numerically stable
        with torch.no_grad():
            self.dt_proj_out.bias.copy_(inv_softplus_dt)
        dt_init_std = dt_rank**-0.5
        nn.init.uniform_(self.dt_proj_out.weight, -dt_init_std, dt_init_std)

    def forward(self, x):
        # x: (B, T, d_inner)
        B_size, T, D = x.shape
        N = self.d_state

        delta = F.softplus(self.dt_proj_out(self.dt_proj_in(x)))  # (B,T,D)
        Bt = self.B_proj(x)  # (B,T,N)
        Ct = self.C_proj(x)  # (B,T,N)
        A = -torch.exp(self.A_log)  # (D,N), strictly negative

        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B,T,D,N)
        deltaB_x = delta.unsqueeze(-1) * Bt.unsqueeze(2) * x.unsqueeze(-1)  # (B,T,D,N)

        h = _parallel_linear_scan(deltaA, deltaB_x)  # (B,T,D,N), h[:,t] = h_t
        y = torch.einsum("btdn,btn->btd", h, Ct)
        return y + x * self.D


class MambaBlock(nn.Module):
    """Full Mamba block as a pre-norm residual layer, as in the reference
    architecture (Gu & Dao, 2023, Fig. 3):

        out = x + out_proj( SSM(SiLU(causal_conv(in_proj_x(norm(x))))) * SiLU(in_proj_z(norm(x))) )

    The pre-LayerNorm and residual are not optional decoration. Without the
    norm, the multiplicative gate `y * SiLU(z)` suppresses small-magnitude
    inputs *quadratically* (SiLU(z) ~ z/2 for small z), and this repo's
    encoder input arrives at std ~0.015 -- the block output collapsed to
    `bias + O(1e-4)`, i.e. effectively input-independent. Verified: logits
    were identical across a whole batch at init (std 0.0000), and training
    then either crawled out of that or got permanently stuck at chance,
    depending on seed/subject. The norm keeps gate inputs O(1); the residual
    gives gradients a direct path around three stacked blocks.
    """

    def __init__(self, d_model, d_state=16, expand=2, d_conv=4):
        super().__init__()
        d_inner = expand * d_model
        self.d_conv = d_conv
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=0)
        self.act = nn.SiLU()
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        h = self.norm(x)
        x_and_z = self.in_proj(h)
        x_branch, z = x_and_z.chunk(2, dim=-1)  # each (B,T,d_inner)

        x_branch = x_branch.transpose(1, 2)  # (B,d_inner,T)
        x_branch = F.pad(x_branch, (self.d_conv - 1, 0))  # left-pad only: causal
        x_branch = self.conv1d(x_branch)  # (B,d_inner,T)
        x_branch = x_branch.transpose(1, 2)  # (B,T,d_inner)
        x_branch = self.act(x_branch)

        y = self.ssm(x_branch)
        y = y * self.act(z)
        return x + self.out_proj(y)

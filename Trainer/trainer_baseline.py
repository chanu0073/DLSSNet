import torch

from Trainer.trainer_corrected_protocol import CorrectedProtocolTrainer


class BaselineTrainer(CorrectedProtocolTrainer):
    """Runs a plain classifier (EEGNet, DeepConvNet, Conformer, MAtt, FBCNet)
    through the exact corrected protocol used for DLSSNet: same 3-way split,
    same S&R augmentation, same early stopping on the session-T validation
    slice, session E touched once. Inherits run() / _eval() / checkpointing /
    resume unchanged -- only the forward pass and the loss differ, because
    these models have no decoder or orthogonality terms.

    `adapt` fixes each model's input convention (they disagree: some expect
    (B, C, T), EEGNet wants (B, 1, C, T), FBCNet wants a 5-D filter bank).
    """

    def __init__(self, model, device, config, adapt=None):
        super().__init__(model, device, config)
        self.adapt = adapt if adapt is not None else (lambda x: x)

    def _forward(self, x):
        out = self.model(self.adapt(x.to(torch.float32)))
        if isinstance(out, tuple):  # some baselines return (features, logits)
            out = out[-1]
        return out, None, None, None

    def _loss(self, out, labels, dec_x, y, basis):
        cls_loss = self.ce(out, labels)
        zero = torch.zeros((), device=cls_loss.device)
        return cls_loss, cls_loss, zero, zero


class FilterBank(torch.nn.Module):
    """9 x 4 Hz band-pass filter bank (4-40 Hz), as FBCNet expects.

    FBCNet is the one baseline that does not take raw EEG: its forward starts
    with `x.permute((0,4,2,3,1))`, i.e. it wants (B, 1, C, T, nBands). The
    bands are applied as fixed FIR kernels via grouped conv1d on the GPU
    rather than scipy.filtfilt on the CPU, so it runs per batch (needed,
    since S&R augmentation regenerates trials every epoch) without becoming
    the bottleneck.
    """

    def __init__(self, n_bands=9, fs=128.0, low=4.0, width=4.0, numtaps=65, multiple_of=4):
        super().__init__()
        from scipy.signal import firwin

        taps = [
            firwin(numtaps, [low + i * width, low + (i + 1) * width], pass_zero=False, fs=fs)
            for i in range(n_bands)
        ]
        self.register_buffer("kernels", torch.tensor(taps, dtype=torch.float32).unsqueeze(1))
        self.pad = numtaps // 2
        self.n_bands = n_bands
        # FBCNet reshapes time into `strideFactor` windows for its log-var
        # layer, so T must divide by it; 438 does not divide by 4, so trim
        # the tail (438 -> 436, i.e. 2 samples / ~16 ms).
        self.multiple_of = multiple_of

    def forward(self, x):
        # (B, C, T) -> (B, 1, C, T', n_bands)
        B, C, T = x.shape
        if self.multiple_of:
            T = T - (T % self.multiple_of)
            x = x[..., :T]
        z = torch.nn.functional.conv1d(
            x.reshape(B * C, 1, T), self.kernels.to(x.dtype), padding=self.pad
        )[..., :T]
        return z.reshape(B, C, self.n_bands, T).permute(0, 1, 3, 2).unsqueeze(1)

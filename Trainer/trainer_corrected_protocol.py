import os
import time

import torch
from torch import nn

from utils.My_loss import reg_loss


class CorrectedProtocolTrainer:
    """
    Trains DLSSNet with a leakage-free protocol:
      - trainloader: session T (minus a validation slice)
      - validloader: held-out slice of session T -- used ONLY for early
        stopping / checkpoint selection
      - testloader: session E -- touched exactly once, after training, to
        report the final accuracy

    Also fixes a gap in Model/DLSSNet/model_zhengjiao.py: Net.forward() only
    returns the *unmasked* embedding (`embeded_x`), not the task-relevance
    masked target the Decoder actually reconstructs against
    (`y = embeded_x * mask_index`), and `mask_index` itself isn't exposed at
    all. This class re-implements the forward pass at the submodule level so
    the reconstruction loss targets the correct, masked signal -- matching
    what the architecture (and the paper's Eq. 5 Loss_rec) specifies. This is
    a read-only wrapper; it does not modify model_zhengjiao.py, so other
    notebooks that rely on its current 9-value return signature are
    unaffected.
    """

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.reg = reg_loss(config.reg_factor)
        self.alpha = config.dec_loss_alpha

    def _forward(self, x):
        m = self.model
        x = x.to(torch.float32)
        x = m.Embedding(x)
        embeded_x = torch.detach(x)
        x, attn, mask_index, pooled_x, scaler, scores = m.Encoder(x)
        y = torch.mul(embeded_x, mask_index)
        out, basis, attn_class, encoded_x = m.classhead(x)
        dec_x, dec_attn = m.Decoder(y, basis)
        return out, dec_x, y, basis

    def _loss(self, out, labels, dec_x, y, basis):
        cls_loss = self.ce(out, labels)
        dec_loss = self.mse(dec_x, y)
        reg_loss_v = self.reg(weight=basis)
        loss = cls_loss + self.alpha * dec_loss + reg_loss_v
        return loss, cls_loss, dec_loss, reg_loss_v

    def _eval(self, loader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                out, *_ = self._forward(imgs)
                pred = out.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def _save_checkpoint(self, ckpt_path, epoch, optimizer, best_val_acc, best_epoch, best_state, epochs_since_improve):
        payload = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "epochs_since_improve": epochs_since_improve,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        # Write to a temp file and rename atomically, so a crash/power-loss
        # mid-write can never leave a truncated, unloadable checkpoint behind.
        tmp_path = ckpt_path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)

    def run(self, trainloader, validloader, testloader, optimizer, ckpt_path=None, writer=None, log_every=20, save_every=100):
        cfg = self.config
        best_val_acc = -1.0
        best_state = None
        best_epoch = 0
        epochs_since_improve = 0
        start_epoch = 1
        t0 = time.time()

        if ckpt_path is not None and os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                start_epoch = ckpt["epoch"] + 1
                best_val_acc = ckpt["best_val_acc"]
                best_epoch = ckpt["best_epoch"]
                best_state = ckpt["best_state"]
                epochs_since_improve = ckpt["epochs_since_improve"]
                torch.set_rng_state(ckpt["rng_state"].cpu())
                if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
                    torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state"]])
                print(
                    f"  resumed from {ckpt_path}: continuing at epoch {start_epoch}, "
                    f"best_val_acc so far={best_val_acc:.4f}@{best_epoch}",
                    flush=True,
                )
            except Exception as e:
                print(f"  WARNING: failed to load checkpoint ({e}); starting fresh from epoch 1", flush=True)
                start_epoch = 1

        for epoch in range(start_epoch, cfg.max_epochs + 1):
            self.model.train()
            for imgs, labels in trainloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                out, dec_x, y, basis = self._forward(imgs)
                loss, cls_loss, dec_loss, reg_loss_v = self._loss(out, labels, dec_x, y, basis)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_acc = self._eval(validloader)
            if writer is not None:
                writer.add_scalar("val_acc", val_acc, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if ckpt_path is not None and epoch % save_every == 0:
                self._save_checkpoint(
                    ckpt_path, epoch, optimizer, best_val_acc, best_epoch, best_state, epochs_since_improve
                )

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"  epoch {epoch}/{cfg.max_epochs} val_acc={val_acc:.4f} "
                    f"best={best_val_acc:.4f}@{best_epoch} elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )

            if epochs_since_improve >= cfg.patience:
                print(
                    f"  early stop at epoch {epoch} (no val improvement for {cfg.patience} epochs)",
                    flush=True,
                )
                break

        self.model.load_state_dict(best_state)
        test_acc = self._eval(testloader)  # touched exactly once
        elapsed = time.time() - t0

        if ckpt_path is not None and os.path.exists(ckpt_path):
            os.remove(ckpt_path)  # completed successfully -- no longer needed for resume

        return dict(
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            test_acc=test_acc,
            elapsed_s=elapsed,
            model_state=best_state,
        )

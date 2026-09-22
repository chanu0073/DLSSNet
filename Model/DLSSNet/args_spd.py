import os

from Model.DLSSNet.args_corrected_protocol import REPO_ROOT, data_config as _base


class data_config(_base):
    """Identical to args_corrected_protocol (same split, losses, augmentation,
    epochs, seeds) -- only the model and the SPD-readout knobs differ, so the
    comparison against the 70.45% augmented baseline is controlled."""

    Experiment_name = "spd_readout"
    model_name = "DLSSNet_SPD"

    # ---- covariance / SPD readout (new) ----
    spd_dim = 20  # BiMap projection k=60 -> 20 before the covariance (210 log-Cholesky features per segment)
    spd_segments = 3  # temporal segments per trial, as in MAtt (m=3)
    spd_eps = 1e-4

    savepath = os.environ.get("DLSSNET_SAVE_PATH", os.path.join(REPO_ROOT, "ExperimentResults_SPD"))
    Result_PATH = savepath
    MODEL_PATH = os.path.join(Result_PATH, "ckpl")
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

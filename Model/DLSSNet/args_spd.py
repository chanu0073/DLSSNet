import os

from Model.DLSSNet.args_corrected_protocol import REPO_ROOT, data_config as _base


class data_config(_base):
    """Identical to args_corrected_protocol (same split, losses, augmentation,
    epochs, seeds) -- only the model and the SPD-readout knobs differ, so the
    comparison against the 70.45% augmented baseline is controlled."""

    Experiment_name = "spd_readout"
    model_name = "DLSSNet_SPD"

    # ---- covariance / SPD readout (new) ----
    # 8, not 20: at 20 the branch contributed 630 features against 1380 from
    # the states -- a lot of extra capacity aimed at 252 training trials, and
    # it cost 6.5 points. 8 gives 36 features/segment, 108 total.
    spd_dim = 8
    spd_segments = 3  # temporal segments per trial, as in MAtt (m=3)
    spd_eps = 1e-4

    savepath = os.environ.get("DLSSNET_SAVE_PATH", os.path.join(REPO_ROOT, "ExperimentResults_SPD"))
    Result_PATH = savepath
    MODEL_PATH = os.path.join(Result_PATH, "ckpl")
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

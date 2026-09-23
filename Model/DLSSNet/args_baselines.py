import os

from Model.DLSSNet.args_corrected_protocol import REPO_ROOT, data_config as _base


class data_config(_base):
    """Same protocol as args_corrected_protocol (3-way split, S&R
    augmentation, epochs, patience, seeds, optimizer, lr) applied to the
    in-repo baselines, so every row of the comparison table is produced
    under identical conditions.

    Note: one shared optimizer/lr for all models rather than each paper's
    own tuned recipe. That is the point -- the table compares architectures
    under one protocol -- but it is a limitation worth stating, since a
    baseline tuned elsewhere may not be at its best here. Per-model
    overrides can be added via the registry in TrainBaselines.py.
    """

    Experiment_name = "baselines_corrected"

    savepath = os.environ.get("DLSSNET_SAVE_PATH", os.path.join(REPO_ROOT, "ExperimentResults_Baselines"))
    Result_PATH = savepath
    MODEL_PATH = os.path.join(Result_PATH, "ckpl")
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

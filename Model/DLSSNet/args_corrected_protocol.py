import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class data_config:
    Experiment_name = "corrected_protocol_baseline"
    model_name = "DLSSNet_corrected"

    # ---- EDIT PER MACHINE (or override via env vars, no code change needed) ----
    data_path = os.environ.get(
        "DLSSNET_DATA_PATH",
        "/home/wangkaixuan/Code/data/BCICIV_2a_mat/",
    )
    gpu_id = int(os.environ.get("DLSSNET_GPU", "0"))
    savepath = os.environ.get(
        "DLSSNET_SAVE_PATH",
        os.path.join(REPO_ROOT, "ExperimentResults_CorrectedProtocol"),
    )

    dataset = "BCICom_2a"
    tasks = {"left": 0, "right": 1, "foot": 2, "tongue": 3}
    num_class = 4
    num_channel = 22
    timepoint = 438

    # Model hyperparameters -- match the paper (Appendix A.1: k=60) and the
    # released Netweights/DLSSNet checkpoints' tensor shapes.
    len_window = 16
    d_model = 60
    frame_stride = 1
    num_frame = timepoint
    num_head = 1
    encoder_num_layers = 1
    low_p = 0.5
    dropout = 0.5
    transformerparwiseforward_dimrat = 1
    statenum = 23

    dec_loss_alpha = 1  # beta in the paper's Eq. 5 (reconstruction loss weight)
    reg_factor = 1  # gamma in the paper's Eq. 5 (orthogonality loss weight)

    subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    bad_subs = None
    ratio = 8  # session-T split: 1/ratio of T goes to validation, rest to train
    seeds = [0, 1, 2]  # multiple seeds -> mean +/- std per subject

    batch_size = 10
    max_epochs = 3000  # full budget; early stopping normally cuts this short
    patience = 300  # epochs without validation improvement before stopping

    lr = 0.0002
    optimizer = "torch.optim.Adam"
    optimizer_parm = {"lr": lr, "betas": (0.5, 0.999)}

    # NOTE: intentionally NOT timestamped. Resuming after a crash/interruption
    # means re-running this same script as a fresh process, and it needs to
    # land on the exact same results.json / checkpoint paths as the run it's
    # continuing -- a per-run timestamp would silently start a new, empty
    # results folder every time and defeat resumability.
    Result_PATH = savepath
    MODEL_PATH = os.path.join(Result_PATH, "ckpl")
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

import os
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class data_config:
    Experiment_name = "mamba_backbone"
    model_name = "DLSSNet_Mamba"

    # ---- EDIT PER MACHINE (or override via env vars, no code change needed) ----
    data_path = os.environ.get(
        "DLSSNET_DATA_PATH",
        "/home/wangkaixuan/Code/data/BCICIV_2a_mat/",
    )
    gpu_id = int(os.environ.get("DLSSNET_GPU", "0"))
    savepath = os.environ.get(
        "DLSSNET_SAVE_PATH",
        os.path.join(REPO_ROOT, "ExperimentResults_Mamba"),
    )

    dataset = "BCICom_2a"
    tasks = {"left": 0, "right": 1, "foot": 2, "tongue": 3}
    num_class = 4
    num_channel = 22
    timepoint = 438

    # Same subject-dependent protocol and shared hyperparameters as
    # args_corrected_protocol.py, so this is a controlled comparison against
    # that 66.94% baseline -- only the backbone (this file's Mamba-specific
    # params below) differs.
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

    dec_loss_alpha = 1
    reg_factor = 1

    # ---- Mamba backbone hyperparameters (new) ----
    ssm_state_dim = 16  # N: per-channel SSM state size
    mamba_expand = 2  # d_inner = expand * d_model
    mamba_conv_kernel = 4  # causal depthwise conv kernel inside each Mamba block
    freq_kernel_sizes = (8, 16, 32)  # multi-branch frequency-aware conv kernel sizes
    tre_hidden = 32  # hidden width of the Dynamic Task-Relevance Estimator's MLP

    subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    bad_subs = None
    ratio = 8
    seeds = [0, 1, 2]

    batch_size = 10
    max_epochs = 3000
    patience = 300

    lr = 0.0002
    optimizer = "torch.optim.Adam"
    optimizer_parm = {"lr": lr, "betas": (0.5, 0.999)}

    Result_PATH = savepath
    MODEL_PATH = os.path.join(Result_PATH, "ckpl")
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

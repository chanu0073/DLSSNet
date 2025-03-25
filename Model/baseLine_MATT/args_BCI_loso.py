import os
import torch
import numpy as np
from datetime import datetime

class data_config:
    Experiment_name = "loso_文超数据"
    
    model_name = "mATT_baseline"

    '''***********- dataset and directory-*************'''
    dataset='BCICom_2a'
    task_class = 'Motor'
    tasks = {"left": 0, "right": 1, "foot": 2, "tongue": 3}

    # tasks = {"left_fist": 0, "right_fist": 1}
    trial_time = 4 #每个试次时长4s
    samplerate = 250
    # timepoint = 438
    # '''***********- model Parameters     -*************'''
    num_channel = 22
    # len_window = 16
    # d_model = 128
    # frame_stride = 5
    # num_frame = int((timepoint-len_window)//frame_stride +1)
    # num_head = 1
    # encoder_num_layers = 2
    # low_p = 0.8
    # dropout = 0.5
    # transformerparwiseforward_dimrat = 4
    # # classhead_dim = 512
    # statenum = 4
    num_class = 4
    data_path = 'C:/2023Experiment/DATA/Data/'

    '''***********- Train Arguments-*************'''    
    nsub = 9
    bad_subs = None
    # bad_subs = np.array([88, 92, 100])
    # train_rate = 0.8
    data_augment = False
    data_augment_inTraining = False
    aug_rate = 1
    num_folds = 9
    
    Time = datetime.now()
    formatedtime = 'ON'+Time.strftime("%Y-%m-%d")+'At'+Time.strftime("%H-%M-%S")
    savepath = 'E:/ExperimentResult/'+Experiment_name
    Result_PATH = os.path.join(savepath, dataset, model_name,formatedtime)
    MODEL_PATH = os.path.join(Result_PATH, 'ckpl')
    LOG_Dir = os.path.join(Result_PATH, 'Log')
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    if not os.path.exists(LOG_Dir):
        os.makedirs(LOG_Dir)
    '''***********- Hyper Arguments-*************'''
    
    # autoaug = 0  # Auto enhancement set to 1
    gpus=[0]  #[1,2,3]
    #WORKERS = 1
    tensorboard= True 
    epochs = 350
    batch_size = 256
    rand_seed=40   #Fixed seed greater than 0
    lr=5e-4
    warm = 0#warm up training phase
    # lr_steps = 80
    # optimizer = "torch.optim.SGD"
    # optimizer_parm = {'lr': 0.001,'momentum':0.9, 'weight_decay':5e-4, 'nesterov':True}
    # optimizer = "torch.optim.AdamW"
    # optimizer_parm = {'lr': 0.001, 'betas': (0.9, 0.999)}
    #学习率：小的学习率收敛慢，但能将loss值降到更低。当使用平方和误差作为成本函数时，随着数据量的增多，学习率应该被设置为相应更小的值。adam一般0.001，sgd0.1，batchsize增大，学习率一般也要增大根号n倍
    #weight_decay:通常1e-4——1e-5，值越大表示正则化越强。数据集大、复杂，模型简单，调小；数据集小模型越复杂，调大。
    # scheduler ="torch.optim.lr_scheduler.MultiStepLR"
    # scheduler_parm ={'milestones':list(range(lr_steps, epochs, lr_steps)), 'gamma':0.8}
    # scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    # scheduler_parm = {'T_max': 60, 'eta_min': 1e-6}
    # scheduler = "torch.optim.lr_scheduler.StepLR"
    # scheduler_parm = {'step_size':1000,'gamma': 0.65}
    # scheduler = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    # scheduler_parm = {'mode': 'min', 'factor': 0.8,'patience':20, 'verbose':True,'threshold':0.0001, 'threshold_mode':'rel', 'cooldown':5, 'min_lr':0, 'eps':1e-08}
    # scheduler = "torch.optim.lr_scheduler.ExponentialLR"
    # scheduler_parm = {'gamma': 0.1}
    loss_f ='torch.nn.CrossEntropyLoss'
    loss_dv = 'torch.nn.KLDivLoss'
    loss_fn = 'torch.nn.BCELoss' # loss_fn = 'torch.nn.BCEWithLogitsLoss'  # loss_fn='torch.nn.MSELoss'
    # fn_weight =[3.734438666137167, 1.0, 1.0, 1.0, 3.5203138607843196, 3.664049338245769, 3.734438666137167, 3.6917943287286734, 1.0, 3.7058695139403963, 1.0, 2.193419513003608, 3.720083373160097, 3.6917943287286734, 3.734438666137167, 1.0, 2.6778551377707998]



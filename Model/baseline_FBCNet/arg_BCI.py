import os
import torch
import numpy as np
from datetime import datetime

class data_config:
    Experiment_name = "BCI2a_matt"
    model_name = "DeepConv"

    '''***********- dataset and directory-*************'''
    dataset='Deep'
    task_class = 'qingxu'
    tasks = {"left": 0, "right": 1, "foot": 2, "tongue": 3}

    # tasks = {"left_fist": 0, "right_fist": 1}
    # trial_time = 4 #每个试次时长4s
    # samplerate = 250
    timepoint = 438
    '''***********- model Parameters     -*************'''
    num_channel = 22
    len_window = 8
    num_class = 4






    data_path = '/home/reid/Experiment/CodesRepository/Data/BCICIV_2a_mat/'


    '''***********- Train Arguments-*************'''    
    nsub = 9
    bad_subs = None
    # bad_subs = np.array([88, 92, 100])
    trainrat = 0.8
    data_augment = False
    data_augment_inTraining = False
    aug_rate = 1
    num_folds = 9
    
    Time = datetime.now()
    # formatedtime = 'ON'+Time.strftime("%Y-%m-%d")+'At'+Time.strftime("%H-%M-%S")
    # savepath = '/media/reid/DATA/experimentLog_TE/'+Experiment_name
    # Result_PATH = os.path.join(savepath, dataset, model_name,formatedtime)
    # MODEL_PATH = os.path.join(Result_PATH, 'ckpl')
    # LOG_Dir = os.path.join(Result_PATH, 'Log')
    # if not os.path.exists(MODEL_PATH):
    #     os.makedirs(MODEL_PATH)
    # if not os.path.exists(LOG_Dir):
    #     os.makedirs(LOG_Dir)
    '''***********- Hyper Arguments-*************'''
    
    # autoaug = 0  # Auto enhancement set to 1
    gpus=[0]  #[1,2,3]
    #WORKERS = 1
    tensorboard= True 
    epochs = 2000
    batch_size = 72
    rand_seed=40   #Fixed seed greater than 0
    lr=0.0002
    warm = 0#warm up training phase
    # lr_steps = 30
    # optimizer = "torch.optim.SGD"
    # optimizer_parm = {'lr': lr,'momentum':0.8, 'weight_decay':5e-4, 'nesterov':False}
    optimizer = "torch.optim.Adam"
    optimizer_parm = {'lr':lr, 'betas':(0.5, 0.999)}
    # optimizer = "torch.optim.AdamW"
    # optimizer_parm = {'lr': 0.002, 'betas': (0.7, 0.999)}
    #学习率：小的学习率收敛慢，但能将loss值降到更低。当使用平方和误差作为成本函数时，随着数据量的增多，学习率应该被设置为相应更小的值。adam一般0.001，sgd0.1，batchsize增大，学习率一般也要增大根号n倍
    #weight_decay:通常1e-4——1e-5，值越大表示正则化越强。数据集大、复杂，模型简单，调小；数据集小模型越复杂，调大。
    scheduler ="torch.optim.lr_scheduler.MultiStepLR"
    scheduler_parm ={'milestones':[epochs], 'gamma':0.5}
    # scheduler = "torch.optim.lr_scheduler.CosineAnnealingLR"
    # scheduler_parm = {'T_max': 200, 'eta_min': 1e-4}
    # scheduler = "torch.optim.lr_scheduler.StepLR"
    # scheduler_parm = {'step_size':1000,'gamma': 0.65}
    # scheduler = "torch.optim.lr_scheduler.ReduceLROnPlateau"
    # scheduler_parm = {'mode': 'min', 'factor': 0.8,'patience':10, 'verbose':True,'threshold':0.0001, 'threshold_mode':'rel', 'cooldown':2, 'min_lr':0, 'eps':1e-08}
    # scheduler = "torch.optim.lr_scheduler.ExponentialLR"
    # scheduler_parm = {'gamma': 0.1}
    loss_f ='torch.nn.CrossEntropyLoss'
    # loss_fn = 'torch.nn.KLDivLoss'
    loss_fn = 'torch.nn.MSELoss' # loss_fn = 'torch.nn.BCEWithLogitsLoss'  # loss_fn='torch.nn.MSELoss'
    # fn_weight =[3.734438666137167, 1.0, 1.0, 1.0, 3.5203138607843196, 3.664049338245769, 3.734438666137167, 3.6917943287286734, 1.0, 3.7058695139403963, 1.0, 2.193419513003608, 3.720083373160097, 3.6917943287286734, 3.734438666137167, 1.0, 2.6778551377707998]


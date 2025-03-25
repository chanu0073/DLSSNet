import numpy as np
import torch
import scipy

def interaug(timg, label, data_config):  
    aug_data = []
    aug_label = []
    timepoint = timg.shape[-1]
    num_frames = int(timepoint/data_config.len_window)
    for cls4aug in range(data_config.num_class):
        cls_idx = np.where(label == cls4aug)
        tmp_data = timg[cls_idx]
        tmp_label = label[cls_idx]
        data_num = tmp_data.shape[0]
        aug_num_eachclass = int(data_num*data_config.aug_rate)
        
        tmp_aug_data = np.zeros((aug_num_eachclass, data_config.num_channel, timepoint))
        for ri in range(aug_num_eachclass):
            for rj in range(num_frames):
                rand_idx = np.random.randint(0, tmp_data.shape[0], num_frames)
                tmp_aug_data[ri, :, rj * data_config.len_window:(rj + 1) * data_config.len_window] = tmp_data[rand_idx[rj], :,
                                                                      rj * data_config.len_window:(rj + 1) * data_config.len_window]

        aug_data.append(tmp_aug_data)
        # np.full(len_data, taskid)
        aug_label.append(np.full(len(tmp_aug_data), cls4aug))
    aug_data = np.concatenate(aug_data)
    aug_label = np.concatenate(aug_label)
    aug_shuffle = np.random.permutation(len(aug_data))
    aug_data = aug_data[aug_shuffle, :, :]
    aug_label = aug_label[aug_shuffle]
    
    return aug_data, aug_label

def get_source_data_physionet(subs, data_config):
    """
    输入文件中数据存储为 {task_name: [样本数量, EEG电极数量, 一个试次采样点]}
    subs: np.array
    data_config.data_path: 数据存放路径 datapath/S001.mat 
    data_config.badsubs: [] 坏的被试数据
    data_config.tasks: {taskname: taskid}
    data_config.train_rate: float, 训练集占总体数据的比例
    data_config.data_augment: True  数据随机帧组合扩增
        如果扩增：
            data_config.aug_rate: 扩增数据比例
            data_config.num_class: 数据类别数量
            data_config.len_window: 帧长度
    """

    TotalsubsTrain = None
    TotalsubsTrainLabel = None
    TotalsubsTest = None
    TotalsubsTestLabel = None
    # train data
    for subn in subs:
        total_data = scipy.io.loadmat(data_config.data_path + 'S'+str(subn).zfill(3)+'.mat')
        data = None
        labels = None
        train_data = None
        test_data = None
        for task, taskid in data_config.tasks.items():
            len_data = len(total_data[task])
            if data is None:
                data = total_data[task]
            else:
                data = np.concatenate((data, total_data[task]), axis=0)
            label = np.full(len_data, taskid)
            if labels is None:
                labels = label
            else:
                labels = np.concatenate((labels, label), axis=0)
        data = data[:, :, :640]
        num_total = len(data)
        shuffle_num = np.random.permutation(num_total)
        data = data[shuffle_num, :, :]
        labels = labels[shuffle_num]
        num_train = int(num_total * data_config.train_rate)
        
        train_data = data[:num_train, :, :]
        train_label = labels[:num_train]
        test_data = data[num_train:, :, :]
        test_label = labels[num_train:]

        target_mean = np.mean(train_data)
        target_std = np.std(train_data)
        train_data = (train_data - target_mean) / target_std
        test_data = (test_data - target_mean) / target_std
        if TotalsubsTrain is None:
            TotalsubsTrain = train_data
            TotalsubsTrainLabel = train_label
            TotalsubsTest = test_data
            TotalsubsTestLabel = test_label
        else:
            TotalsubsTrain = np.concatenate((TotalsubsTrain, train_data), axis=0)
            TotalsubsTrainLabel = np.concatenate((TotalsubsTrainLabel,train_label), axis=0)
            TotalsubsTest = np.concatenate((TotalsubsTest, test_data), axis=0)
            TotalsubsTestLabel = np.concatenate((TotalsubsTestLabel, test_label), axis=0)

    if data_config.data_augment:
        augdata, auglabels = interaug(TotalsubsTrain, TotalsubsTrainLabel, data_config)
        TotalsubsTrain = np.concatenate((TotalsubsTrain, augdata), axis = 0)
        TotalsubsTrainLabel = np.concatenate((TotalsubsTrainLabel, auglabels), axis = 0)
    
    
    return TotalsubsTrain, TotalsubsTrainLabel, TotalsubsTest, TotalsubsTestLabel

def get_source_data_BCICom2A(subs, data_config):
    """
    
    """
    pass

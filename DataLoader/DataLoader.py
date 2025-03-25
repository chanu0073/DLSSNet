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
        train_data = None
        test_data = None
        train_label = None
        test_label = None
        for task, taskid in data_config.tasks.items():
            len_data = len(total_data[task])
            data = total_data[task][:,:,:640]
            shuffle_num = np.random.permutation(len_data)
            data = data[shuffle_num, :]
            
            num_train = int(len_data * data_config.train_rate)
            t_data = data[:num_train, :]
            v_data = data[num_train+1:, :]
            t_label = np.full(len(t_data), taskid)
            v_label = np.full(len(v_data), taskid)
            if train_data is None:
                train_data = t_data
                test_data = v_data
                train_label = t_label
                test_label = v_label
            else:
                train_data = np.concatenate((train_data, t_data), axis=0)
                test_data = np.concatenate((test_data, v_data), axis=0)
                train_label = np.concatenate((train_label, t_label), axis=0)
                test_label = np.concatenate((test_label, v_label), axis=0)
            
        shuffle_num_t = np.random.permutation(len(train_data))
        shuffle_num_v = np.random.permutation(len(test_data))
        
        
        train_data = train_data[shuffle_num_t, :]
        train_label = train_label[shuffle_num_t]
        test_data = test_data[shuffle_num_v, :]
        test_label = test_label[shuffle_num_v]

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

def get_source_data(data_path, data_config):

    total_data = scipy.io.loadmat(data_path)
    data_n = []
    labels = []
    for task, task_id in data_config.tasks.items():
        data = total_data[task][:,:,:(data_config.trial_time*data_config.samplerate)]
        data_n.append(data)
        label = np.full(len(data), task_id)
        labels.append(label)
    data_n = np.concatenate(data_n, axis=0)
    labels = np.concatenate(labels, axis = 0)
    shuffle_num_t = np.random.permutation(len(data_n))
    data_n = data_n[shuffle_num_t, :, :]
    labels = labels[shuffle_num_t]
    return data_n, labels

def data_Normalization(train_data, test_data):
    target_mean = np.mean(train_data)
    target_std = np.std(train_data)
    test_mean = np.mean(test_data)
    test_std = np.std(test_data)
    train_data = (train_data - target_mean) / target_std
    test_data = (test_data - test_mean) / test_std

    return train_data, test_data

def data_split2traintest(total_data, total_label, data_config):
    shuffle_num_t = np.random.permutation(len(total_data))
    total_data = total_data[shuffle_num_t, :, :]
    total_label = total_label[shuffle_num_t]

    train_num = int(len(total_data)*data_config.train_rate)
    train_data = total_data[:train_num, :, :]
    train_label = total_label[:train_num]
    test_data = total_data[train_num+1:, :,:]
    test_label = total_label[train_num+1:]

    return train_data, train_label, test_data, test_label


def get_source_data_train_val(data_path, data_config):

    total_data = scipy.io.loadmat(data_path)
    T_data_n = []
    T_labels = []
    v_data_n = []
    v_labels = []
    for task, task_id in data_config.tasks.items():
        data = total_data[task][:,:,:(data_config.trial_time*data_config.samplerate)]
        label = np.full(len(data), task_id)
        shuffle_num = np.random.permutation(len(data))
        data = data[shuffle_num, :, :]
        label = label[shuffle_num]
        T_data_n.append(data[0:int(data_config.trainrat*len(data)), :, :])
        T_labels.append(label[0:int(data_config.trainrat*len(data))])
        v_data_n.append(data[int(data_config.trainrat*len(data)):, :, :])
        v_labels.append(label[int(data_config.trainrat*len(data)):])

    T_data_n = np.concatenate(T_data_n, axis=0)
    T_labels = np.concatenate(T_labels, axis=0)
    v_data_n = np.concatenate(v_data_n, axis=0)
    v_labels = np.concatenate(v_labels, axis=0)

    shuffle_num_t = np.random.permutation(len(T_data_n))
    shuffle_num_v = np.random.permutation(len(v_data_n))
    T_data_n = T_data_n[shuffle_num_t, :, :]
    T_labels = T_labels[shuffle_num_t]
    v_data_n = v_data_n[shuffle_num_v, :, :]
    v_labels = v_labels[shuffle_num_v]

    return T_data_n, T_labels, v_data_n, v_labels
        
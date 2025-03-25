import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne


def get_weight(att):
    x_ticks = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
               'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
               'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
               'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

    eeg_weight = {}

    for i in range(len(x_ticks)):
        eeg_weight[x_ticks[i]] = np.sum(att, axis=0)[i] - 0.5
        print(eeg_weight[x_ticks[i]])
    # 重构自定义矩阵顺序
    reWeight = []
    for key in chLa_index:
        val = eeg_weight[key]
        reWeight.append(val)

    return reWeight

def softmax(x):
    """ softmax function """
    # x -= np.min(x, axis=0, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), keepdims=True)

    return x

def plot_eeg_topomap(eeg_weight, eeg_info):
    im, cn = mne.viz.plot_topomap(eeg_weight,
                                  eeg_info,
                                  names=chLa_index.tolist(),
                                  vlim=(0, 1),
                                  size=5,
                                  cmap='RdBu_r',
                                  show=False)
    plt.colorbar(im)
    plt.show()


builtin_montages = mne.channels.get_builtin_montages(descriptions=True)
int32_montage = mne.channels.make_standard_montage("biosemi32")

# weight = {'Fp1': 0.31, 'AF3': -0.26, 'F7': 1.22, 'F3': 0.99, 'Fz': 0.71, 'FC5': 0.55, 'FC1': -1.19, 'T7': 0.61, 'C3': -0.80, 'Cz': 2.36, 'CP5': -0.74, 'CP1': 0.72, 'P7': 0.93, 'P3': 0.38, 'Pz': 1.07, 'PO3': -1.46, 'O1': -0.12, 'Oz': 1.074, 'Fp2': 1.04, 'AF4': -0.065, 'F4': -0.52, 'F8': 0.37, 'FC2': 1.30, 'FC6': 0.94, 'C4': -1.11, 'T8': -0.16, 'CP2': 1.82, 'CP6': 0.41, 'P4': 0.46, 'P8': 0.99, 'PO4':0.15, 'O2':0.23}

# 查看脑地形图矩阵导联位置
sensor_data_32 = int32_montage.get_positions()['ch_pos']
sensor_dataframe_32 = pd.DataFrame(sensor_data_32).T
chLa_index = sensor_dataframe_32.index.values

info = mne.create_info(
        ch_names=list(chLa_index),
        ch_types=['eeg']*32,   # 通道个数
        sfreq=1000)            # 采样频率

info.set_montage(int32_montage)

atts = np.load('./deap_4_atts.npy')

for i in range(len(atts)):
    att = get_weight(atts[i])
    plot_eeg_topomap(att, info)







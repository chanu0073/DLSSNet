import torch
import torch.utils.data as Data
from scipy import io
import numpy as np
import os

def split_train_valid_set(x_train, y_train, ratio):
    s = y_train.argsort()
    x_train = x_train[s]
    y_train = y_train[s]

    cL = int(len(x_train) / 4)

    class1_x = x_train[ 0 * cL : 1 * cL ]
    class2_x = x_train[ 1 * cL : 2 * cL ]
    class3_x = x_train[ 2 * cL : 3 * cL ]
    class4_x = x_train[ 3 * cL : 4 * cL ]

    class1_y = y_train[ 0 * cL : 1 * cL ]
    class2_y = y_train[ 1 * cL : 2 * cL ]
    class3_y = y_train[ 2 * cL : 3 * cL ]
    class4_y = y_train[ 3 * cL : 4 * cL ]

    vL = int(len(class1_x) / ratio)

    x_train = torch.cat((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]))
    y_train = torch.cat((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]))

    x_valid = torch.cat((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]))
    y_valid = torch.cat((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]))

    return x_train, y_train, x_valid, y_valid



# split dataset
def getAllDataloader(subject, ratio, data_path, bs):
    train = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_T.mat'))
    test = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_E.mat'))
    x_train = torch.Tensor(train['x_train'])
    y_train = torch.Tensor(train['y_train']).view(-1)
    x_test = torch.Tensor(test['x_test'])
    y_test = torch.Tensor(test['y_test']).view(-1)

    x_train, y_train, x_valid, y_valid = split_train_valid_set(x_train, y_train, ratio=ratio)
    dev = torch.device('cpu')

    x_train = x_train[:, :, 124:562].to(dev)
    y_train = y_train.long().to(dev)
    x_valid = x_valid[:, :, 124:562].to(dev)
    y_valid = y_valid.long().to(dev)
    x_test = x_test[:, :, 124:562].to(dev)
    y_test = y_test.long().to(dev)
    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)
    print(x_test.shape)
    print(y_test.shape)
    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)
    test_dataset = Data.TensorDataset(x_test, y_test)

    trainloader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = bs,
        shuffle = True,
        num_workers = 0,
        pin_memory=True
    )
    validloader = Data.DataLoader(
        dataset = valid_dataset,
        batch_size = 2,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )
    testloader =  Data.DataLoader(
        dataset = test_dataset,
        batch_size = 2,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )

    return trainloader, validloader, testloader
        
def getAllDataloader_withouttest(subject, ratio, data_path, bs):
    file_t = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_T.mat'))
    file_e = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_E.mat'))
    # x_t = torch.Tensor(file_t['x_train'])
    # y_t = torch.Tensor(file_t['y_train']).view(-1)
    # x_e = torch.Tensor(file_e['x_test'])
    # y_e = torch.Tensor(file_e['y_test']).view(-1)

    x_t = torch.Tensor(file_t['x_train'])
    y_t = torch.Tensor(file_t['y_train']).view(-1)
    x_e = torch.Tensor(file_e['x_test'])
    y_e = torch.Tensor(file_e['y_test']).view(-1)

    x_all = torch.concatenate([x_t, x_e], dim = 0)
    y_all = torch.concatenate([y_t, y_e], dim = 0)



    x_train, y_train, x_valid, y_valid = split_train_valid_set(x_all, y_all, ratio=ratio)
    dev = torch.device('cpu')

    x_train = x_train[:, :, 124:562].to(dev)
    y_train = y_train.long().to(dev)
    x_valid = x_valid[:, :, 124:562].to(dev)
    y_valid = y_valid.long().to(dev)

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)

    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)


    trainloader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = bs,
        shuffle = True,
        num_workers = 0,
        pin_memory=True
    )
    validloader = Data.DataLoader(
        dataset = valid_dataset,
        batch_size = 2,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )

    return trainloader, validloader


def getAllDataloader_toT_E(subject, ratio, data_path, bs):
    file_t = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_T.mat'))
    file_e = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_E.mat'))
    # x_t = torch.Tensor(file_t['x_train'])
    # y_t = torch.Tensor(file_t['y_train']).view(-1)
    # x_e = torch.Tensor(file_e['x_test'])
    # y_e = torch.Tensor(file_e['y_test']).view(-1)

    x_t = torch.Tensor(file_t['x_train'])
    y_t = torch.Tensor(file_t['y_train']).view(-1)
    x_e = torch.Tensor(file_e['x_test'])
    y_e = torch.Tensor(file_e['y_test']).view(-1)


    dev = torch.device('cpu')

    x_train = x_t[:, :, 124:562].to(dev)
    y_train = y_t.long().to(dev)
    x_valid = x_e[:, :, 124:562].to(dev)
    y_valid = y_e.long().to(dev)

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)

    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)


    trainloader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = bs,
        shuffle = True,
        num_workers = 0,
        pin_memory=True
    )
    validloader = Data.DataLoader(
        dataset = valid_dataset,
        batch_size = 2,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )

    return trainloader, validloader

def getCVFolds(subject, n_folds, data_path, bs, val_bs=2):
    """Stratified k-fold cross-validation folds from session T ONLY.

    For hyperparameter tuning. A single 36-trial validation split is far too
    noisy to choose between configurations: at ~75% accuracy its binomial
    standard error is 7.2 points, and measured on this repo's own runs the
    within-subject correlation between validation and test accuracy is
    r=+0.11 (p=0.60) -- i.e. picking the best-validation run picks noise.
    Averaging over k folds uses all 288 session-T trials for the decision and
    cuts that standard error to 2.6 points.

    Session E is NEVER loaded here, by construction -- the held-out test set
    cannot leak into a tuning decision if this function cannot read it.

    With n_folds=8 each fold is 252 train / 36 validation, exactly matching
    the split sizes of the reporting protocol in getAllDataloader(), so
    hyperparameters chosen here transfer to the final runs.

    Returns a list of (trainloader, validloader), one per fold.
    """
    train = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_T.mat'))
    x = torch.Tensor(train['x_train'])[:, :, 124:562]
    y = torch.Tensor(train['y_train']).view(-1).long()

    # Stratified: split each class's trials into n_folds contiguous chunks,
    # so every fold holds the same class balance as the whole session.
    per_class_chunks = []
    for c in torch.unique(y):
        idx = torch.nonzero(y == c).flatten()
        per_class_chunks.append(torch.chunk(idx, n_folds))

    folds = []
    for f in range(n_folds):
        val_idx = torch.cat([chunks[f] for chunks in per_class_chunks])
        mask = torch.ones(len(y), dtype=torch.bool)
        mask[val_idx] = False
        tr_idx = torch.nonzero(mask).flatten()

        trainloader = Data.DataLoader(
            dataset=Data.TensorDataset(x[tr_idx], y[tr_idx]),
            batch_size=bs, shuffle=True, num_workers=0, pin_memory=True,
        )
        validloader = Data.DataLoader(
            dataset=Data.TensorDataset(x[val_idx], y[val_idx]),
            batch_size=val_bs, shuffle=False, num_workers=0, pin_memory=True,
        )
        folds.append((trainloader, validloader))
    return folds


def getDeep(data_path, ratio, bs):
    x = np.load(os.path.join(data_path, 'data.npy'))
    y = np.load(os.path.join(data_path, '4classes_label.npy'))
    y = np.argmax(y, axis=1)

    
    x = torch.Tensor(x)
    y = torch.Tensor(y).view(-1)
    # x_e = torch.Tensor(file_e['x_test'])
    # y_e = torch.Tensor(file_e['y_test']).view(-1)
    x_train, y_train, x_valid, y_valid = split_train_valid_set(x, y, ratio=ratio)
    
    dev = torch.device('cpu')

    x_train = x_train.to(dev)
    y_train = y_train.long().to(dev)
    x_valid =x_valid.to(dev)
    y_valid = y_valid.long().to(dev)

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)

    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)


    trainloader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = bs,
        shuffle = True,
        num_workers = 0,
        pin_memory=True
    )
    validloader = Data.DataLoader(
        dataset = valid_dataset,
        batch_size = bs,
        shuffle = False,
        num_workers = 0,
        pin_memory=True
    )

    return trainloader, validloader
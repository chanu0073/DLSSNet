import os
import warnings
import tqdm.auto as tqdm

import torch
from torch import softmax
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from Model.GPoolingwithdecoderForAnalysis3.args_zjBCIsingletrain_val import data_config
# from DataLoader.DataLoader import get_source_data_train_val, data_Normalization, interaug
from DataLoader.GetBci2a import getAllDataloader_withouttest
from Metrics.metrics import AverageMeter, accuracy
from Model.GPoolingwithdecoderForAnalysis3.model_zhengjiao import Net
from utils.arg_utils import *
from Trainer.trainermodel_decoder_zhengjiaoyueshu_single_withouttest import trainer
import gc

if __name__ == '__main__':
    '''***********- Hyper Arguments-*************'''
    # warnings.filterwarnings("ignore")
    # device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device=torch.device(device_name)
    subs = np.arange(1, data_config.nsub+1)
    subs = np.setdiff1d(subs, data_config.bad_subs)
    subs_ran = subs
    folds = np.array_split(subs_ran, data_config.num_folds)
    torch.cuda.set_device(data_config.gpus[0])
    # folds = np.array([[8], [9]])
    All_ACC = AverageMeter()
    All_writer = SummaryWriter(log_dir=data_config.LOG_Dir)
    
    if data_config.rand_seed > 0:
        init_rand_seed(data_config.rand_seed)
    configFilepath = data_config.Result_PATH + '/data_config_summary.txt'
    with open(configFilepath, 'w') as f:
        f.write("***********- data_config Parameters -*************\n")
        for key, value in vars(data_config).items():
            f.write(f"{key}: {value}\n")
        f.write("***********- End of data_config Parameters -*************\n")



    
    

    fold_n = 1
    for subid in subs_ran:
        print("***********- ***********- READ DATA and processing-*************")
        log_dir = os.path.join(data_config.LOG_Dir, "Fold"+str(fold_n).zfill(2))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

   
        model_path = os.path.join(data_config.MODEL_PATH, "Fold"+str(fold_n).zfill(2))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        writer = SummaryWriter(log_dir=log_dir)

        trainloader, validloader = getAllDataloader_withouttest(subject=subid, 
                                                            ratio=8, 
                                                            data_path=data_config.data_path, 
                                                            bs=data_config.batch_size)

        
        # trainloader, validloader, testloader = getAllDataloader(subject=subid, 
        #                                                     ratio=8, 
        #                                                     data_path=data_config.data_path, 
        #                                                     bs=data_config.batch_size)
      # x, y = train_dataset[0]
        # print('input:', x.size(), 'lable:', y)  # [3, 32, 32]) 6

        print("***********- loading model -*************")
        model = Net(num_channels=data_config.num_channel, 
                    len_window=data_config.len_window, 
                    d_model=data_config.d_model, 
                    frame_stride=data_config.frame_stride,
                    num_frame=data_config.num_frame,
                    num_head=data_config.num_head, 
                    encoder_num_layers=data_config.encoder_num_layers, 
                    low_p = data_config.low_p,
                    dropout=data_config.dropout, 
                    transformerparwiseforward_dimrat=data_config.transformerparwiseforward_dimrat, 
                    statenum=data_config.statenum,
                    num_class=data_config.num_class)

        model = model.cuda()
        model_total_para = sum([param.nelement() for param in model.parameters()])
        print("Number of parameter: %.2fM" % (model_total_para/1e6))
        
        
        optimizer = eval(data_config.optimizer)(
                                                model.parameters(), **data_config.optimizer_parm)
        scheduler = eval(data_config.scheduler)(
                                                optimizer, **data_config.scheduler_parm)
        loss_f = eval(data_config.loss_f)()
        # loss_dv = eval(data_config.loss_dv)()
        loss_fn = eval(data_config.loss_fn)()


        
        Trainer = trainer(loss_f, loss_fn, model, optimizer, scheduler, config=data_config, writer=writer)
        
        best_acc, top_score = Trainer.run(trainloader, validloader, model_path)
        All_writer.add_scalar(tag = 'Sub_best_val_acc', scalar_value=best_acc, global_step=fold_n)
        
        All_ACC.update(best_acc)

        fold_n = fold_n+1
        writer.close()
        
        foldConfigFilePath = log_dir+'config_Summary.txt'
        with open(foldConfigFilePath, 'w') as f:
            f.write(str(subid)+'\n')
            f.write("Number of parameter: %.2fM \n" % (model_total_para/1e6))
            f.write("***********- Top_score -*************\n")
            f.write(np.array2string(top_score)+"\n")
            f.write("***********- Besttop-1 val_accuracy -*************\n")
            f.write(str(best_acc)+"\n")


        
        del trainloader, validloader
        gc.collect()

    print("所有被试_验证集_最佳准确率平均值为:"+str(All_ACC.avg))
    with open(configFilepath, 'a') as f:
        f.write("所有被试_验证集_最佳准确率平均值为:"+str(All_ACC.avg)+"\n")

    All_writer.close()
        




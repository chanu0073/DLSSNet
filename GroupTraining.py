import os
import warnings
import tqdm.auto as tqdm

import torch
from torch import softmax
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from Model.DLSSNet.args_multiStatenumzjBCIsingletrain_val import data_config
# from DataLoader.DataLoader import get_source_data_train_val, data_Normalization, interaug
from DataLoader.GetBci2a import getAllDataloader_withouttest
from Metrics.metrics import AverageMeter, accuracy
from Model.DLSSNet.model_zhengjiao import Net
from utils.arg_utils import *
from Trainer.trainermodel_multistatenum import trainer
import gc

if __name__ == '__main__':
    '''***********- Hyper Arguments-*************'''
    # warnings.filterwarnings("ignore")
    # device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device=torch.device(device_name)
    subs = data_config.subs
    subs = np.setdiff1d(subs, data_config.bad_subs)
    subs_ran = subs
    folds = np.array_split(subs_ran, data_config.num_folds)

    # folds = np.array([[8], [9]])
    All_ACC = AverageMeter()
    
    torch.cuda.set_device(data_config.gpus[0])
    if data_config.rand_seed > 0:
        init_rand_seed(data_config.rand_seed)
    configFilepath = data_config.Result_PATH + '/data_config_summary.txt'
    with open(configFilepath, 'w') as f:
        f.write("***********- data_config Parameters -*************\n")
        for key, value in vars(data_config).items():
            f.write(f"{key}: {value}\n")
        f.write("***********- End of data_config Parameters -*************\n")
        f.write("***********- Subs splited into 10folds -*************\n")
        for fold_s in folds:
            i = 1
            f.write(f"Folds{i}TestSub_id:{np.array2string(fold_s)}\n")
            i = i+1



    
    
    All_summary = []
    All_summary_sorted = []
    All_summary_acc = []
    for subid in subs_ran:
        print("***********- ***********- READ DATA and processing-*************")


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
        print("***********- ***********- setup The Experiment-*************")
        sub_basis_acc_dec = np.zeros([len(data_config.statenum), 4], dtype=float)
        
        sub_log_dir = os.path.join(data_config.LOG_Dir, "subject"+str(subid).zfill(2))
        if not os.path.exists(sub_log_dir):
            os.makedirs(sub_log_dir)
        sub_All_writer = SummaryWriter(log_dir=sub_log_dir)


        for iter_state in range(len(data_config.statenum)):
            model = None
            state_num = data_config.statenum[iter_state]
            
            

            log_dir = os.path.join(data_config.LOG_Dir, "subject"+str(subid).zfill(2), "State_num"+str(state_num).zfill(2))
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            model_path = os.path.join(data_config.MODEL_PATH, "subject"+str(subid).zfill(2), "State_num"+str(state_num).zfill(2))
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        
            writer = SummaryWriter(log_dir=log_dir)
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
                    statenum=state_num,
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


        
            Trainer = trainer(loss_f, loss_fn, model, optimizer, scheduler, config=data_config, writer=writer, basis_num=state_num, sub_id=subid)
        
            best_acc, train_decloss, val_decloss, top_score = Trainer.run(trainloader, validloader, model_path)
            
            sub_basis_acc_dec[iter_state] = [state_num, best_acc, train_decloss, val_decloss]
            
            

            sub_All_writer.add_scalar(tag = 'Sub'+str(subid)+'_numbasis_best_val_acc', scalar_value=best_acc, global_step=iter_state)
            
            Sub_state_Result_FilePath = log_dir+'result_Summary.txt'
            with open(Sub_state_Result_FilePath, 'w') as f:
                f.write(str(subid)+':'+str(state_num)+'\n')
                f.write("Number of parameter: %.2fM \n" % (model_total_para/1e6))
                f.write("***********- Top_score -*************\n")
                f.write(np.array2string(top_score)+"\n")
                f.write("***********- Besttop-1 val_accuracy -*************\n")
                f.write(str(best_acc)+"\n")

        z = np.argsort(-sub_basis_acc_dec[:, 1])
        sub_basis_acc_dec_sorted = sub_basis_acc_dec[z]
        sub_best_acc = sub_basis_acc_dec[z][0, 1]
        sub_All_writer.close()
        
        Sub_Result_FilePath = sub_log_dir+'result_Summary.txt'
        with open(Sub_Result_FilePath, 'w') as f:
            f.write(str(subid)+'\n')
            f.write("***********- Besttop-1 val_accuracy -*************\n")
            f.write(str(sub_best_acc)+"\n")
            f.write("***********- Best in basisnums -*************\n")
            f.write(np.array2string(sub_basis_acc_dec)+"\n")
            f.write("***********- Best in basisnums_sorted -*************\n")
            f.write(np.array2string(sub_basis_acc_dec_sorted)+"\n")
            

        All_summary.append(sub_basis_acc_dec)
        All_summary_sorted.append(sub_basis_acc_dec_sorted)
        All_summary_acc.append(sub_best_acc)
        
        All_ACC.update(sub_best_acc)
        
        writer.close()

            




        
        del trainloader, validloader
        gc.collect()
    
    ResultFilepath = data_config.Result_PATH + '/Result_summary.txt'
    with open(ResultFilepath, 'w') as f:
        f.write("所有被试_验证集_最佳准确率平均值为:"+str(All_ACC.avg)+"\n")
        for i in range(len(subs_ran)):
            subid = subs_ran[i]
            f.write(f"subject"+str(subid).zfill(2)+"\n")
            f.write("最佳准确率："+str(All_summary_acc[i])+"\n")
            f.write("Summary:\n"+np.array2string(All_summary[i])+"\n")
            f.write("Summary_sorted:\n"+np.array2string(All_summary_sorted[i])+"\n")

    
    
    print("所有被试_验证集_最佳准确率平均值为:"+str(All_ACC.avg))
    
        




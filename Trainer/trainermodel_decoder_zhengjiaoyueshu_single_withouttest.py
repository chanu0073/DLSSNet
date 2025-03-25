import numpy as np
from tqdm.auto import tqdm
import torch
from torch import softmax
from Metrics.metrics import AverageMeter
from utils.data_utils import save_checkpoint
import torchmetrics.functional as tf
from DataLoader.DataLoader import interaug
from torch.autograd import Variable
from utils.My_loss import reg_loss



class trainer:
    def __init__(self, loss_f, loss_fn, model, optimizer, scheduler, writer, config):
        self.loss_f = loss_f
        self.loss_fn = loss_fn
        self.reg_loss = reg_loss(config.reg_factor)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.writer = writer
        self.alpha = config.dec_loss_alpha

    def batch_train(self, batch_imgs, batch_labels, epoch):
        
        attns, predicted, dec_x, y, states = self.model(batch_imgs)
        loss, cls_loss, dec_loss, reg_loss = self.myloss(predicted, batch_labels, dec_x, y, states)
        predicted = softmax(predicted, dim=-1)
        del batch_imgs, batch_labels
        return loss, predicted, attns, cls_loss, dec_loss, reg_loss

    def train_epoch(self, loader,  epoch):
        self.model.train()
        tqdm_loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        cls_losses = AverageMeter()
        dec_losses = AverageMeter()
        reg_losses = AverageMeter()

        print("\n************Training*************")
        for batch_idx, (imgs, labels) in enumerate(tqdm_loader):
            # print("data",imgs.size(), labels.size())#[128, 3, 32, 32]) torch.Size([128]
            if self.config.data_augment_inTraining:
                augdata = None
                auglabel = None
                # img = Variable(img.cuda().type(self.Tensor))
                # label = Variable(label.cuda().type(self.LongTensor))
                augdata, auglabel = interaug(imgs, labels, self.config)
                augdata = torch.from_numpy(augdata)
                augdata = augdata.float()
                auglabel = torch.from_numpy(auglabel)
                auglabel = auglabel.long()
                
                imgs = torch.cat((imgs, augdata))
                labels = torch.cat((labels, auglabel))

            if (len(self.config.gpus) > 0):
                imgs, labels = imgs.cuda(), labels.cuda()
                    
            # print(self.optimizer.param_groups[0]['lr'])
            loss, predicted, attns, cls_loss, dec_loss, reg_loss = self.batch_train(imgs, labels, epoch)
            losses.update(loss.item(), imgs.size(0))
            cls_losses.update(cls_loss.item(), imgs.size(0))
            dec_losses.update(dec_loss.item(), imgs.size(0))
            reg_losses.update(reg_loss.item(), imgs.size(0))
            # print(predicted.size(),labels.size())

            self.optimizer.zero_grad()
            loss.backward()
            # if epoch == 1 or epoch == self.config.epochs:
            #     for name, param in self.model.named_parameters():
            #         if not param.grad is None:
            #             self.writer.add_histogram(tag=name+'_grad', values=param.grad.clone().cpu().numpy(), global_step=epoch)
            #             self.writer.add_histogram(tag=name+'_data', values=param.data.clone().cpu().numpy(), global_step=epoch)
            self.optimizer.step()
            

            acc1 = tf.accuracy(predicted.data, labels, task = 'multiclass', num_classes=self.config.num_class, average='micro')
            top1.update(acc1.item(), imgs.size(0))
            

            tqdm_loader.set_description('Training: loss:{:.4}/{:.4} lr:{:.4} acc1:{:.4} clsloss:{:.4}/{:.4} decloss:{:.4}/{:.4} regloss:{:.4}/{:.4}'.
                                        format(loss, losses.avg, self.optimizer.param_groups[0]['lr'], top1.avg, cls_loss, cls_losses.avg,dec_loss, dec_losses.avg, reg_loss, reg_losses.avg))
            # if batch_idx%1==0:
            #     break
        return top1.avg, losses.avg, cls_losses.avg, dec_losses.avg, reg_losses.avg

    def valid_epoch(self, loader, epoch):
        self.model.eval()
        # tqdm_loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        cls_losses = AverageMeter()
        dec_losses = AverageMeter()
        reg_losses = AverageMeter()

        print("\n************Evaluation*************")
        for batch_idx, (batch_imgs, batch_labels) in enumerate(loader):
            with torch.no_grad():
                if (len(self.config.gpus) > 0):
                    batch_imgs, batch_labels = batch_imgs.cuda(), batch_labels.cuda()
                attns, predicted, dec_x, y, states = self.model(batch_imgs)
                loss, cls_loss, dec_loss, reg_loss = self.myloss(predicted, batch_labels,dec_x, y,states)
                loss = loss.detach().cpu().numpy()
                cls_loss = cls_loss.detach().cpu().numpy()
                dec_loss = dec_loss.detach().cpu().numpy()
                reg_loss = reg_loss.detach().cpu().numpy()
                predicted = softmax(predicted, dim=-1)
                losses.update(loss.item(), batch_imgs.size(0))
                cls_losses.update(cls_loss.item(), batch_imgs.size(0))
                dec_losses.update(dec_loss.item(), batch_imgs.size(0))
                reg_losses.update(reg_loss.item(), batch_imgs.size(0))
                acc1 = tf.accuracy(predicted.data, batch_labels, task = 'multiclass', num_classes=self.config.num_class,average='micro')
                top1.update(acc1.item(), batch_imgs.size(0))

        return top1.avg, losses.avg, cls_losses.avg, dec_losses.avg, reg_losses.avg
    
    

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.config.lr
        if self.config.dataset.startswith('cifar'):
            # lr = self.config.lr * (0.1 ** (epoch // (self.config.epochs * 0.3))) * (0.1 ** (epoch // (self.config.epochs * 0.75)))
            if epoch < 60:
                lr = self.config.lr
            elif epoch < 120:
                lr = self.config.lr * 0.2
            elif epoch < 160:
                lr = self.config.lr * 0.04
            else:
                lr = self.config.lr * 0.008
        elif self.config.dataset == ('imagenet'):
            if self.config.epochs == 300:
                lr = self.config.lr * (0.1 ** (epoch // 75))
            else:
                lr = self.config.lr * (0.1 ** (epoch // 30))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



    def myloss(self, predicted, labels,dec_x, y, states):
        # print(predicted.size(),labels.size())#[128, 10]) torch.Size([128])
        cls_loss = self.loss_f(predicted, labels)
        dec_loss = self.loss_fn(dec_x, y)
        reg_loss = self.reg_loss(weight = states)
        loss = cls_loss+self.alpha * dec_loss + reg_loss
        return loss, cls_loss, dec_loss, reg_loss

    def run(self, train_loder, val_loder, model_path):
        best_acc1= 0
        start_epoch = 0
        top_score = np.zeros([5, 3], dtype=float)
        iter_per_epoch = len(train_loder)
        # warmup_scheduler = WarmUpLR(self.optimizer, iter_per_epoch * self.config.warm)
        
        # model, optimizer, start_epoch=load_checkpoint(self.model,self.optimizer,model_path)
        for e in range(self.config.epochs):
            e = e+start_epoch+1
            print("------model:{}----Epoch: {}--------".format(self.config.model_name, e))
            
                # adjust_learning_rate(self.optimizer,e,self.config.model_type)
            # torch.cuda.empty_cache()
            train_acc, train_loss, train_cls_loss, train_dec_loss, train_reg_loss = self.train_epoch(train_loder, e)
            acc1, val_loss,val_cls_loss, val_dec_loss, val_reg_loss= self.valid_epoch(val_loder, e)
            self.scheduler.step()
            #
            print("\nval_loss:{:.4f} | acc1:{:.4f} | cls_loss:{:.4f} dec_loss{:.4f} reg_loss{:.4f} ".format(val_loss, acc1, val_cls_loss, val_dec_loss, val_reg_loss))

            # if acc1 >= best_acc1:
            #     best_acc1 = acc1
            #     print('Current Best (top-1 error):', best_acc1)
            # if err5 <= best_err5:
            #     best_err5 = err5
            #     print('Current Best (top-5 error):', best_err5)

            if acc1 > top_score[4][2]:
                top_score[4] = [e, val_loss, acc1]
                z = np.argsort(-top_score[:, 2])
                top_score = top_score[z]
                best_acc1 = save_checkpoint(self.model, self.optimizer, e, train_losses=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=acc1, check_acc=best_acc1,
                                            savepath=model_path, m_name=self.config.model_name)
                # print("\ntest_loss:{:.4f} | test_acc1:{:.4f} ".format(test_loss,testacc1))
                
                
            #     top_score5[4] = err5
            #     z = np.argsort(top_score5)
            #     top_score5 = top_score5[z]
            #     # print(top_score5)
            if (self.config.tensorboard):
                self.writer.add_scalar('current best acc', best_acc1,e)
                self.writer.add_scalar('training loss', train_loss, e)
                self.writer.add_scalar('valing loss', val_loss, e)
                self.writer.add_scalar('acc1', acc1, e)
                self.writer.add_scalar('training cls_loss', train_cls_loss, e)
                self.writer.add_scalar('training dec_loss', train_dec_loss, e)
                self.writer.add_scalar('training reg_loss', train_reg_loss, e)
                self.writer.add_scalar('valing cls_loss', val_cls_loss, e)
                self.writer.add_scalar('valing dec_loss', val_dec_loss, e)
                self.writer.add_scalar('valing reg_loss', val_reg_loss, e)
                

                # for i in range(len(train_attns)):
                #     self.writer.add_image(tag=str(i)+"layer_train_attn", img_tensor=train_attns[i][1, :].clone().cpu().numpy(), global_step=e, dataformats="HW")
                # for i in range(len(val_attns)):
                #     self.writer.add_histogram(tag=str(i)+"layer_test_attn", img_tensor=val_attns[i][1, :].clone().cpu().numpy(), global_step=e, dataformat)

                # self.writer.add_scalar('err5', err5, e)

        
        print('\nbest score:{}'.format(self.config.model_name))
        for i in range(5):
            print(top_score[i])
        # print(top_score5, top_score[:, 0])
        print('Besttop-1 accuracy):',
              top_score[:, 1].mean(), best_acc1)

        print("best accuracy:\n avg_acc1:{:.4f} | best_acc1:{:.4f} ".
              format(top_score[:, 2].mean(), best_acc1))
                
        self.writer.close()
        return best_acc1, top_score
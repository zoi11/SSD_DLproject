
import torch
from torch.optim.lr_scheduler import MultiStepLR
from Data import Our_Dataloader
from .structs import multiboxloss
from Utils.visdom_op import visdom_line, setup_visdom, visdom_bar
from torch import nn
from torch.nn import DataParallel
import os

__all__ = ['Trainer']

class Trainer(object):
    
    
    def __init__(self, cfg, max_iter=None, batch_size=None, num_workers = None, train_devices=None,
                 model_save_step=None, model_save_root=None, vis = None, vis_step=None):
        
        self.cfg = cfg

        self.iterations = self.cfg.TRAIN.MAX_ITER
        if max_iter:
            self.iterations = max_iter

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        if batch_size:
            self.batch_size = batch_size

        self.num_workers = cfg.TRAIN.NUM_WORKERS
        if num_workers:
            self.num_workers = num_workers

        self.train_devices = cfg.DEVICE.TRAIN_DEVICES
        if train_devices:
            self.train_devices = train_devices

        self.model_save_root = cfg.FILE.MODEL_SAVE_ROOT
        if model_save_root:
            self.model_save_root = model_save_root

        if not os.path.exists(self.model_save_root):
            os.mkdir(self.model_save_root)
        self.model_save_step = self.cfg.STEP.MODEL_SAVE_STEP
        if model_save_step:
            self.model_save_step = model_save_step

        self.vis = setup_visdom()
        if vis:
            self.vis = vis
        self.vis_step = self.cfg.STEP.VIS_STEP
        if vis_step:
            self.vis_step = vis_step

        self.model = None
        self.loss_func = None
        self.optimizer = None
        self.scheduler = None

    def __call__(self, model, dataset):
       
        if not isinstance(model, nn.DataParallel):
            
            model = DataParallel(model, device_ids=self.train_devices)
        self.model = model
        data_loader = Our_Dataloader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print(' Max_iter = {}, Batch_size = {}'.format(self.iterations, self.batch_size))
        print(' Model will train on cuda:{}'.format(self.train_devices))

        num_gpu_use = len(self.train_devices)
        if (self.batch_size % num_gpu_use) != 0:
            raise ValueError(
                'You use {} gpu to train , but set batch_size={}'.format(num_gpu_use, data_loader.batch_size))

        self.set_lossfunc()
        self.set_optimizer()
        self.set_scheduler()

        print("Set optimizer : {}".format(self.optimizer))
        print("Set scheduler : {}".format(self.scheduler))
        print("Set lossfunc : {}".format(self.loss_func))


        print(' Start Train......')
        print(' -------' * 20)

        for iteration, (images, boxes, labels, image_names) in enumerate(data_loader):
            iteration+=1
            boxes, labels = boxes.to('cuda'), labels.to('cuda')
            cls_logits, bbox_preds = self.model(images)
            reg_loss, cls_loss = self.loss_func(cls_logits, bbox_preds, labels, boxes)

            reg_loss = reg_loss.mean()
            cls_loss = cls_loss.mean()
            loss = reg_loss + cls_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']

            if iteration % 10 == 0:
                print('Iter : {}/{} | Lr : {} | Loss : {:.4f} | cls_loss : {:.4f} | reg_loss : {:.4f}'.format(iteration, self.iterations, lr, loss.item(), cls_loss.item(), reg_loss.item()))

            if self.vis and iteration % self.vis_step == 0:
                visdom_line(self.vis, y=[loss], x=iteration, win_name='loss')
                visdom_line(self.vis, y=[reg_loss], x=iteration, win_name='reg_loss')
                visdom_line(self.vis, y=[cls_loss], x=iteration, win_name='cls_loss')
                visdom_line(self.vis, y=[lr], x=iteration, win_name='lr')

            if iteration % self.model_save_step == 0:
                torch.save(model.module.state_dict(), '{}/model_{}.pkl'.format(self.model_save_root, iteration))
                
            if iteration == self.iterations:
                torch.save(model.module.state_dict(), '{}/model_{}.pkl'.format(self.model_save_root, iteration))
                return True
        return True

    def set_optimizer(self, lr=None, momentum=None, weight_decay=None):
     
        if not lr:
            lr= self.cfg.OPTIM.LR
        if not momentum:
            momentum = self.cfg.OPTIM.MOMENTUM
        if not weight_decay:
            weight_decay = self.cfg.OPTIM.WEIGHT_DECAY

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)

    def set_lossfunc(self, neg_pos_ratio=None):
       
        if not neg_pos_ratio:
            neg_pos_ratio = self.cfg.TRAIN.NEG_POS_RATIO
        self.loss_func = multiboxloss(neg_pos_ratio=neg_pos_ratio)
        # print(' Trainer set loss_func : {}, neg_pos_ratio = {}'.format('multiboxloss', neg_pos_ratio))

    def set_scheduler(self, lr_steps=None, gamma=None):
       
        if not lr_steps:
            lr_steps = self.cfg.OPTIM.SCHEDULER.LR_STEPS
        if not gamma:
            gamma = self.cfg.OPTIM.SCHEDULER.GAMMA
        self.scheduler = MultiStepLR(optimizer=self.optimizer,
                                     milestones=lr_steps,
                                     gamma=gamma)
        # print(' Trainer set scheduler : {}, lr_steps={}, gamma={}'.format('MultiStepLR', lr_steps, gamma))

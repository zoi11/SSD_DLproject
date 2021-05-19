

from Model import SSD, Trainer
from Data import VOCDataset
from Configs import _C as cfg
from Data import SSDTramsfrom,SSDTargetTransform


# training dataset
train_dataset=VOCDataset(cfg, is_train=True, transform=SSDTramsfrom(cfg,is_train=True),
                         target_transform=SSDTargetTransform(cfg))

# test dataset
test_dataset = VOCDataset(cfg=cfg, is_train=False,
                          transform=SSDTramsfrom(cfg=cfg, is_train=False),
                          target_transform=SSDTargetTransform(cfg))

if __name__ == '__main__':


    # building network
    net = SSD(cfg)
    
    net.to(cfg.DEVICE.MAINDEVICE)

    # initialize trainer
    trainer = Trainer(cfg,max_iter=10000)

    print(trainer.optimizer)
    # start training
    trainer(net, train_dataset)
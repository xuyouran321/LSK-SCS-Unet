from geoseg.models.ablation_network import res_convssm
from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.uavid_dataset import *
from geoseg.models.ablation_network.res_convssm import Resconv
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 100
ignore_index = 255
train_batch_size = 8
val_batch_size = 8
lr = 1e-3
weight_decay = 0.01
backbone_lr = lr
backbone_weight_decay = weight_decay
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "res_convssm"
weights_path = "model_weights/uavid/{}".format(weights_name)
test_weights_name = "res_convssm_test"#拼接路径
log_name = 'uavid/{}'.format(weights_name)#log记录
monitor = 'val_mIoU' #目标函数
monitor_mode = 'max'#选择best model
save_top_k = 1 #保留唯一模型
save_last = True #best ckpt，last ckpt
check_val_every_n_epoch = 1 #每轮都验证
pretrained_ckpt_path = None # 无预训练，从头运行
# gpus = ['auto']  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
gpus = [0]
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = Resconv(num_classes=num_classes)
# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)

use_aux_loss = False

# define the dataloader

train_dataset = UAVIDDataset(data_root='data/uavid/train_val', img_dir='images', mask_dir='masks',
                             mode='train', mosaic_ratio=0.25, transform=train_aug, img_size=(1024, 1024))

val_dataset = UAVIDDataset(data_root='data/uavid/val_val', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=val_aug, img_size=(1024, 1024))


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=2,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)


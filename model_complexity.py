import torch
import numpy as np

from thop import profile
import time
from types import SimpleNamespace
from geoseg.models.ABCNet import ABCNet
from geoseg.models.BANet import BANet
from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.DCSwin import dcswin_tiny
from geoseg.models.UNetFormer_lsk_all import UNetFormer_lsk_t
from geoseg.models.A2FPN import A2FPN
from geoseg.models.lsk_convssm import lsk_convssm
from geoseg.models.FTUNetFormer import FTUNetFormer

def calculate(net):
    x = torch.randn(2, 3, 256, 256).cuda()
    net.cuda()
    net.eval()
    # print(net.flops())
    out = net(x)
    flops, params = profile(net, (x,))
    print(flops / 1e9)  # 打印 GFLOPS
    print(params / 1e6) # 打印参数量，单位是百万
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # 转换为 MB
    print("Max memory allocated:", max_memory_allocated, "MB")
    # 重置最大内存计数
    torch.cuda.reset_max_memory_allocated()

    x = torch.zeros((1,3,256,256)).cuda()
    t_all = []
    for i in range(1000):
        t1 = time.time()
        y = net(x)
        t2 = time.time()
        t_all.append(t2 - t1)
    print('average fps:',1 / np.mean(t_all)) #速度 (FPS)

# print("=====ABCNet======")
# abc_net = ABCNet(n_classes=8)
# calculate(abc_net)


# print("=====UNetFormer======")
# unetformer = UNetFormer(num_classes=8)
# calculate(unetformer)

# print("=====dcswin_tiny======")
# dcswin = dcswin_tiny(num_classes=8)
# calculate(dcswin)

# print("=====all======")
# all = UNetFormer_lsk_t(num_classes=8)
# calculate(all)

# print("=====BaNet======")
# BaNet = BANet(num_classes=8)
# calculate(BaNet)

# print("=====A2FPN======")
# a2FPN = A2FPN()
# calculate(a2FPN)

print("=====lsk_convssm======")
Lsk_convssm = lsk_convssm()
calculate(Lsk_convssm)

# print("=====FTUNetFormer======")
# FTUNetformer = FTUNetFormer()
# calculate(FTUNetformer)
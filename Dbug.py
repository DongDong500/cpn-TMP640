from torchsummary import summary
import torch

import models
from get_argparser import get_argparser

if __name__ == "__main__":
    args = get_argparser().parse_args()

    net = models.models.__dict__['backbone_resnet101'](args, )
    
    inputs = torch.rand(5, 3, 256, 256)
    print(summary(net, (3, 256, 256), device='cpu'))
    print(net(inputs).shape)
    #print(net.parameters())
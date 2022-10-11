import os

import torch.utils.data as data
from torch.utils.data import DataLoader

from PIL import Image

from . import ext_transforms as et
from .peroneal import Peroneal


def get_dst(args, ):

    dst = Peroneal()

    return dst

def get_loader(args, ):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = et.ExtCompose([

        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std),
        ])
    val_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std),
        ])
    test_transform = et.ExtCompose([
        et.ExtToTensor(),
        et.ExtNormalize(mean=mean, std=std),
        ])

    if args.run_test:
        test_dst = get_dst(args, )
        test_loader = DataLoader(test_dst, batch_size=args.test_batch_size, 
                                    num_workers=args.num_workers, shuffle=True, drop_last=True)
        loader = test_loader
    else:
        train_dst = get_dst(args, )
        val_dst = get_dst(args, )
        train_loader = DataLoader(train_dst, batch_size=args.train_batch_size, 
                                    num_workers=args.num_workers, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dst, batch_size=args.val_batch_size, 
                                    num_workers=args.num_workers, shuffle=True, drop_last=True)
        loader = [train_loader, val_loader]

    return loader
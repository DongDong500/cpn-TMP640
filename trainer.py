import random

import numpy as np

import torch
import torch.nn as nn

from datetime import datetime

import models
import utils

def set_optim(args, model_name, model):
    ### Optimizer
    if model_name.startswith("deeplab"):
        if args.optim == "SGD":
            optimizer = torch.optim.SGD(params=[
            {'params': model.encoder.parameters(), 'lr': 0.1 * args.lr},
            {'params': model.decoder.parameters(), 'lr': args.lr},
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim == "RMSprop":
            optimizer = torch.optim.RMSprop(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr},
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim == "Adam":
            optimizer = torch.optim.Adam(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr},
            ], lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        else:
            raise NotImplementedError
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=5e-4, weight_decay=5e-4, momentum=0.9)
    
    ### Scheduler
    if args.lr_policy == 'lambdaLR':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, )
    elif args.lr_policy == 'multiplicativeLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, )
    elif args.lr_policy == 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=args.step_size)
    elif args.lr_policy == 'multiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, )
    elif args.lr_policy == 'exponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, )
    elif args.lr_policy == 'cosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, )
    elif args.lr_policy == 'cyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer=optimizer, )
    else:
        raise NotImplementedError

    return optimizer, scheduler

def train_epoch():
    ...

def val_epoch():
    ...

def run_training(args, ) -> dict:
    start_time = datetime.now()
    results = {
        "Short Memo" : args.short_memo + " Kfold-" + str(args.kfold),
        "F1 score" : {
            "background" : 0.00,
            "RoI" : 0.00
        },
        "Bbox regression" : {
            "MSE" : 0.00,
        },
        "time elapsed" : "h:m:s.ms"
    }

    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    ### Load datasets
    loader = utils.get_loader(args, )

    ### Load model
    model = models.models.__dict__[args.model]()

    ### Set up optimizer and scheduler
    optim, sched = set_optim(args, args.model, model)

    ### Resume models, schedulers and optimizer
    if args.resume:
        raise NotImplementedError
    else:
        print("[!] Train from scratch...")
        resume_epoch = 0
    
    if torch.cuda.device_count() > 1:
        print('cuda multiple GPUs')
        model = nn.DataParallel(model)

    model.to(devices)

    ### Set up metrics

    ### Train


    results['time elapsed'] = str(datetime.now() - start_time)

    return results
import os
import random

import numpy as np

import torch
import torch.nn as nn

from datetime import datetime

import models
import utils

def set_optim(args, model):

    ### Optimizer
    if args.model.startswith("deeplab"):
        if args.optim == "SGD":
            optimizer = torch.optim.SGD(params=[
            {'params': model.encoder.parameters(), 'lr': 0.1 * args.lr},
            {'params': model.decoder.parameters(), 'lr': args.lr},
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
        elif args.optim == "RMSprop":
            optimizer = torch.optim.RMSprop(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr},
            ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay )
        elif args.optim == "Adam":
            optimizer = torch.optim.Adam(params=[
            {'params': model.backbone.parameters(), 'lr': 0.1 * args.lr},
            {'params': model.classifier.parameters(), 'lr': args.lr},
            ], lr=args.lr, betas=(0.9, 0.999), eps=1e-8 )
        else:
            raise NotImplementedError
    else:
        if args.optim == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum )
        elif args.optim == "RMSprop":
            optimizer = torch.optim.RMSprop(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum )
        elif args.optim == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum )
        else:
            raise NotImplementedError
    
    ### Scheduler
    if args.lr_policy == 'lambdaLR':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, )
    elif args.lr_policy == 'multiplicativeLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, )
    elif args.lr_policy == 'stepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=args.step_size )
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

def train_epoch(devices, model, loader, optimizer, scheduler, metrics, ):

    model.train()
    metrics.reset()
    running_loss = 0.0

    crietrion = ...

    for i, (ims, lbls) in enumerate(loader):
        optimizer.zero_grad()

        ims = ims.to(devices)
        lbls = lbls.to(devices)

        outputs = model(ims)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1].detach().cpu().numpy()
        true = lbls.detach().cpu().numpy()
        
        loss = crietrion(outputs, lbls)
        loss.backward()
        optimizer.step()
        metrics.update(true, preds)

        running_loss += loss.item() * ims.size(0)
    scheduler.step()
    epoch_loss = running_loss / len(loader)
    score = metrics.get_results()

    return epoch_loss, score

def val_epoch(devices, model, loader, metrics, ):

    model.eval()
    metrics.reset()
    running_loss = 0.0

    crietrion = ...

    with torch.no_grad():
        for i, (ims, lbls) in enumerate(loader):

            ims = ims.to(devices)
            lbls = lbls.to(devices)

            outputs = model(ims)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1].detach().cpu().numpy()
            true = lbls.detach().cpu().numpy()
            
            loss = crietrion(outputs, lbls)
            metrics.update(true, preds)

            running_loss += loss.item() * ims.size(0)
        epoch_loss = running_loss / len(loader)
        score = metrics.get_results()

    return epoch_loss, score

def run_training(args, RUN_ID, DATA_FOLD) -> dict:
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
    optimizer, scheduler = set_optim(args, model)

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
    metrics = utils.StreamSegMetrics(n_classes=2)
    early_stop = utils.EarlyStopping(patience=args.patience, delta=args.delta, verbose=True, 
                    path=os.path.join(args.BP_pth, RUN_ID, DATA_FOLD), ceiling=True, )
    
    ### Train
    for epoch in range(resume_epoch, args.total_itrs):
        epoch_loss, score = train_epoch(devices, model, loader[0], optimizer, scheduler, metrics,)

        epoch_loss, score = val_epoch(devices, model, loader[1], metrics, )

        if early_stop.early_stop:
            print("Early Stop !!!")
            break

        if args.run_demo and epoch > 2:
            print("Run Demo !!!")
            break
    
    results['time elapsed'] = str(datetime.now() - start_time)

    return results
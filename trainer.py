import os
import random

import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime

import criterion
import models
import utils

def print_result(phase, score, epoch, total_itrs, loss):
    print("[{}] Epoch: {}/{} Loss: {:.5f}".format(phase, epoch, total_itrs, loss))
    print("\tF1 [0]: {:.5f} [1]: {:.5f}".format(score['Class F1'][0], score['Class F1'][1]))
    print("\tIoU[0]: {:.5f} [1]: {:.5f}".format(score['Class IoU'][0], score['Class IoU'][1]))
    print("\tOverall Acc: {:.3f}, Mean Acc: {:.3f}".format(score['Overall Acc'], score['Mean Acc']))

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

def train_epoch(devices, model, loader, optimizer, scheduler, metrics, args):

    model.train()
    metrics.reset()
    running_loss = 0.0

    crietrion = criterion.get_criterion.__dict__[args.loss_type]()

    for i, (ims, lbls) in tqdm(enumerate(loader)):
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

def val_epoch(devices, model, loader, metrics, args):

    model.eval()
    metrics.reset()
    running_loss = 0.0

    crietrion = criterion.get_criterion.__dict__[args.loss_type]()

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
        epoch_loss, score = train_epoch(devices, model, loader[0], optimizer, scheduler, metrics, args)
        print_result('train', score, epoch, args.total_itrs, epoch_loss)
        epoch_loss, score = val_epoch(devices, model, loader[1], metrics, args)
        print_result('val', score, epoch, args.total_itrs, epoch_loss)

        if early_stop(score['Class F1'][1], model, optimizer, scheduler, epoch):
            best_score = score
            best_loss = epoch_loss

        if early_stop.early_stop:
            print("Early Stop !!!")
            break

        if args.run_demo and epoch >= 2:
            print("Run Demo !!!")
            break
    
    results['F1 score']['background'] = best_score['Class F1'][0]
    results['F1 score']['RoI'] = best_score['Class F1'][1]
    results['time elapsed'] = str(datetime.now() - start_time)

    return results
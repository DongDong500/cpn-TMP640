import argparse

import models

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--short_memo", type=str, default='short memo', 
                        help="")
    parser.add_argument("--login", type=str, default='/home/dongik/src/login.json', 
                        help="SMTP login ID&PW")
    parser.add_argument("--current_time", type=str, default='current_time', 
                        help="")
    parser.add_argument("--prefix", type=str, default='/', 
                        help="path prefix")
    parser.add_argument("--TB_pth", type=str, default='/', 
                        help="tensorboard log")
    parser.add_argument("--BP_pth", type=str, default='/', 
                        help="best param")
    parser.add_argument("--data_pth", type=str, default='/home/dongik/datasets',
                        help="")
    parser.add_argument("--random_seed", type=int, default=1, 
                        help="random seed (default: 1)")
    parser.add_argument("--exp_itrs", type=int, default=1, 
                        help='repeat n identical experiments (default: 1)')
    # Model options
    available_models = sorted(name for name in models.models.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              models.models.__dict__[name]) )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50', choices=available_models,
                        help='model name (default: deeplabv3plus_resnet50)')
    
    # Dataset options
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="number of workers (default: 8)")
    parser.add_argument("--modality", type=str, default="UN", 
                        help='UN (unknown), HM (HM70A) or SN (miniSONO) (default: UN)')
    parser.add_argument("--region", type=str, default="peroneal", 
                        help='peroneal, median-forearm or median-wrist (default: peroneal)')
    parser.add_argument("--kfold", type=int, default=5, 
                        help="kfold (default: 5)")
    parser.add_argument("--k", type=int, default=0, 
                        help="i-th fold set of kfold data (default: 0)")
    parser.add_argument("--train_batch_size", type=int, default=32, 
                        help='train batch size (default: 32)')
    parser.add_argument("--val_batch_size", type=int, default=16, 
                        help='validate batch size (default: 16)') 
    parser.add_argument("--test_batch_size", type=int, default=16, 
                        help='test batch size (default: 16)')

    # Train options
    parser.add_argument("--total_itrs", type=int, default=1600,
                        help="epoch number (default: 1.6k)")
    parser.add_argument("--loss_type", type=str, default='entropydice',
                        help="criterion (default: ce+dl)")
    parser.add_argument("--optim", type=str, default='SGD',
                        help="optimizer (default: SGD)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument("--lr", type=float, default=1e-1,
                        help="learning rate (default: 1e-1)")
    parser.add_argument("--lr_policy", type=str, default='stepLR',
                        help="scheduler")
    parser.add_argument("--step_size", type=int, default=100, 
                        help="step size (default: 100)")
    
    # Early-stop options
    parser.add_argument("--patience", type=int, default=100,
                        help="Number of epochs with no improvement after which training will be stopped (default: 100)")
    parser.add_argument("--delta", type=float, default=0.001,
                        help="Minimum change in the monitored quantity to qualify as an improvement (default: 0.001)")

    # Resume model from checkpoint
    parser.add_argument("--resume", action='store_true',
                        help="resume from checkpoint (defaults: false)")
    parser.add_argument("--resume_ckpt", default='/', type=str,
                        help="resume from checkpoint (defalut: /)")
    parser.add_argument("--continue_training", action='store_true',
                        help="restore state from reserved params (defaults: false)")

    parser.add_argument("--run_test", action='store_true', 
                        help='inference')
    parser.add_argument("--run_demo", action='store_true',
                        help='')
    
    return parser
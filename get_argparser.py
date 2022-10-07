import argparse

def get_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mail_pth", type=str, default='/home/dongik/src/login.json', help="smtp login-info pth")

    parser.add_argument("--current_time", type=str, default='current_time', help="name of log folder (default: current_time)")
    parser.add_argument("--dir_prefix", type=str, default='/', help="prefix of dir")
    parser.add_argument("--tensorboard_dir", type=str, default='/', help="tensorboard log dir")
    parser.add_argument("--best_param_dir", type=str, default='/', help="best param dir")
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    
    parser.add_argument("--train_itrs", type=int, default=1, help='repeat n identical experiments (default: 1)')

    parser.add_argument("--data_dir", type=str, default='/home/dongik/datasets', help="data directory folder")
    parser.add_argument("--dataset", type=str, default="", help='primary dataset (default: )')
    parser.add_argument("--data_fold", type=int, default=5, help="k-fold of data (default: 5)")
    parser.add_argument("--kfold", type=int, default=0, help="i-th fold of k-fold data (default: 0)")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers (default: 8)")
    
    parser.add_argument("--train_batch_size", type=int, default=32, help='train batch size (default: 32)')
    parser.add_argument("--val_batch_size", type=int, default=16, help='validate batch size (default: 16)') 
    parser.add_argument("--test_batch_size", type=int, default=16, help='test batch size (default: 16)')

    parser.add_argument("--test-mode", action='store_true')
    parser.add_argument("--run_demo", action='store_true')
    
    return parser
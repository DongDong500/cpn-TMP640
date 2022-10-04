import os
import socket
import argparse

from datetime import datetime

import utils

def main_worker(args, ):
    params = utils.save_argparser(args, args.log_dir_fdr)

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--smtp_dir", type=str, default='/home/dongik/src/login.json',
                        help="G-mail smtp login")
    parser.add_argument("--log_dir_fdr", type=str, default='/',
                        help="log directory folder")
    parser.add_argument("--current_time", type=str, default='current_time',
                        help="name of log folder (default: current_time)")
    
    parser.add_argument("--data_dir_fdr", type=str, default='/home/dongik/datasets',
                        help="data directory folder")
    parser.add_argument("--dataset", type=str, default="cpn_vit", choices=available_datasets,
                        help='primary dataset (default: cpn_vit)')   
    parser.add_argument("--dataset_ver", type=str, default="splits/v5/3",
                        help="version of primary dataset (default: splits/v5/3)")
                                          
    parser.add_argument("--run_demo", action='store_true')
    
    return parser

def _mkdir(args, verbose=False):

    hostname = socket.gethostname()
    s_folder = os.path.dirname( os.path.abspath(__file__) ).split('/')[-1] + '-result'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') + ('_demo' if args.run_demo else '')
    if verbose:
        print(f'hostname: {hostname}\nfolder: {s_folder}\ncurrent time: {current_time}')
    
    args.current_time = current_time
    args.log_dir_prefix = os.path.join()


def main():
    total_time = datetime.now()

    try:
        parser = get_argparser().parse_args()
        
        _mkdir(parser, verbose=False, ) 
        main_worker(parser, )

    except KeyboardInterrupt:
        ...
    except Exception as e:
        ...
    
    total_time = datetime.now() - total_time
    print('Time elapsed (h:m:s.ms) {}'.format(total_time))



if __name__ == "__main__":
    main(verbose=False)
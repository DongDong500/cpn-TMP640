import os
import socket

from datetime import datetime
import traceback

import utils
from trainer import run_training
from get_argparser import get_argparser

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

def mkLogDir(args, verbose=False):

    hostname = socket.gethostname()
    rdir = os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )
    s_folder = os.path.dirname( os.path.abspath(__file__) ).split('/')[-1] + '-result'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') + ('_demo' if args.run_demo else '')

    if verbose:
        print(f'hostname: {hostname}\nfolder: {s_folder}\ncurrent time: {current_time}\ndir prefix: {rdir}')
    
    args.current_time = current_time
    args.dir_prefix = os.path.join(rdir, s_folder, current_time)
    args.tensorboard_dir = os.path.join(rdir, s_folder, current_time, 'tensorboard')
    args.best_param_dir = os.path.join(rdir, s_folder, current_time, 'best-param')

    if not os.path.exists(args.dir_prefix):
        os.makedirs(args.dir_prefix)
        os.makedirs(os.path.join(args.dir_prefix, 'best-param'))
        os.makedirs(os.path.join(args.dir_prefix, 'tensorboard'))


def main_worker(args, ) -> dict:
 
    params = utils.save_argparser(args, args.dir_prefix)
  
    results = {}
    for train_itrs in range(0, args.train_itrs):
        RUN_ID = 'run_' + str(train_itrs).zfill(2)
        print(f"{train_itrs + 1}-th iteration")
        
        results[RUN_ID] = {}
        for data_fold in range(0, args.data_fold):
            args.kfold = data_fold
            DATA_FOLD = 'fold_' + str(data_fold).zfill(2)
            print(f"{data_fold + 1}-th data fold")

            os.makedirs(os.path.join( args.tensorboard_dir, RUN_ID, DATA_FOLD ))
            os.makedirs(os.path.join( args.best_param_dir, RUN_ID, DATA_FOLD ))

            results[RUN_ID][DATA_FOLD] = run_training(args)
            
            utils.Email()

    return results


def main():
    total_time = datetime.now()

    try:
        is_error = False

        args = get_argparser().parse_args()
        
        mkLogDir(args, verbose=True, )

        result = main_worker(args, )
        utils.save_dict_to_json(d= ,json_path=)

    except KeyboardInterrupt:
        is_error = True
        print("KeyboardInterrupt: Stop !!!")

    except Exception as e:
        is_error = True
        print("Error: ", e)
        print(traceback.format_exc())
    
    if is_error:
        os.rename(args.log_dir_fdr, args.log_dir_fdr + '_aborted')

    total_time = datetime.now() - total_time
    print('Time elapsed (h:m:s.ms) {}'.format(total_time))
    

if __name__ == "__main__":
    main()
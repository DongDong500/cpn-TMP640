import random

import numpy as np

import torch

import models
import utils

def run_training(args, ) -> dict:
    '''
        return (dict): {
            "Short Memo" : "str...",
            "F1 score" : {
                "background" : 0.98,
                "RoI" : 0.85
            },
            "Bbox regression" : {
                "MSE" : 0.05,
            },
            "time elapsed" : "h:m:s.ms"
        }
    '''
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    ### Load datasets
    loader = utils.get_loader(args, )

    ### Load model
    model = models.load_model(args, )

    ### Set up optimizer and scheduler

    ### Resume models, schedulers and optimizer

    ### Set up metrics

    

    return results
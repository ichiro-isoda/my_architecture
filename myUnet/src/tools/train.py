from argparse import ArgumentParser
import torch
import time
import os
import sys
import numpy as np
import random
from torch.utils.data import DataLoader

sys.path.append(os.pardir)
from utils.parser import create_dataset_parser, create_model_parser, create_runtime_parser
from utils.utils import config_read, get_model, createOpbase
from utils.optimizer import set_optimizer
from src.lib.dataset import get_dataset
from src.lib.trainer import SegTrainer
seed = 116

def main():
    # ========================================
    #  read config file and set each options 
    # ========================================
    ap = ArgumentParser(description='train U-Net for segmentation')
    ap.add_argument('--conf_file','-c',default='confs/example.cfg',help='Specify the config file path')
    base_args, remaining_argv = ap.parse_known_args()
    args = config_read(base_args.conf_file, remaining_argv, 
                       ["Dataset","Model","Runtime"], [create_dataset_parser, create_model_parser, create_runtime_parser])
    #seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ============================
    #  prepare datasete iterator 
    # ============================
    print('Loading datasets...')
    train_dataset, validation_dataset = get_dataset(args)
    train_iterator = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch,
        shuffle=True
    )
    validation_iterator = DataLoader(
        dataset=validation_dataset,
        batch_size=args.batch,
        shuffle=False
    )

    # initialize training model
    model = get_model(args)
    if args.init_model is not None:
        print('Load model from', args.init_model)
        model = torch.load(args.init_model)
    model = model.to(args.gpu)

    # ==========================================================
    #    prepare file and directory for saving trained models
    # ==========================================================
    opbase = createOpbase(args.save_dir)
    os.makedirs(os.path.join(opbase, 'trained_models'), exist_ok=True)
    psep = '/'
    print('-- train_dataset.size = {}\n-- validation_dataset.size = {}'.format(
        len(train_dataset), len(validation_dataset)))
    print(args, file=args.savedir+'/condition.txt')
    
    # prepare optimizer
    optimizer = set_optimizer(args, model)

    trainer = SegTrainer(
        model=model, optimizer=optimizer, 
        train_data=train_iterator, test_data=validation_iterator, 
        epoch=args.epoch, device=args.gpu, loss_func=eval(args.loss_func), validation=args.validation
    )

    start_time = time.time()
    ####   training   ####
    best_score = trainer.training((train_iterator, validation_iterator))
    end_time = time.time()

    process_time = end_time - start_time
    print('Elapsed time is (sec) {}'.format(process_time))
    print('best validation loss is {}'.format(best_score))

if __name__ == '__main__':
    main()


import logging
import os
import argparse

import coloredlogs

from Coach import Coach
from mastergoal.MastergoalGame import MastergoalGame as Game
from mastergoal.NNet import NNetWrapper as nn
from utils import *

# Debug, trying to reproduce a specific error
import torch
import random
import numpy as np
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info. #OR INFO

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 30,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 100, #18496
    'arenaCompare': 10,
    'cpuct': 1,

    'checkpoint': './IDEAL/',
    'load_model': False,
    'load_folder_file': ('./05_02_25', 'checkpoint_1.pth.tar'),
    'starting_iteration': 1,
    'numItersForTrainExamplesHistory': 40,
    'verbose': True,

    # New parameters for Google Drive backup
    'use_drive_backup': True,
    'drive_backup_path': '/content/drive/My Drive/BACKUP_FOLDER'
})

def main():
    # Google Drive mounting code
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_drive_backup', action='store_true')
    parser.add_argument('--drive_backup_path', type=str)
    parsed_args = parser.parse_args()

    # Merge parsed arguments into the existing 'args' object.
    args.use_drive_backup = parsed_args.use_drive_backup
    args.drive_backup_path = parsed_args.drive_backup_path

    logging.basicConfig(level=logging.INFO)
    logging.info(f"use_drive_backup: {args.use_drive_backup}")
    logging.info(f"drive_backup_path: {args.drive_backup_path}")

    try:
        if 'google.colab' in str(get_ipython()):
            from google.colab import drive
            logging.info('Running in Colab environment. Mounting Google Drive...')
            drive.mount('/content/drive')
            logging.info('Google Drive mounted successfully.')
        else:
            logging.warning('Not running in Colab environment.')
            args.use_drive_backup = False
    except Exception as e:
        logging.error(f'Failed to mount Google Drive: {e}')
        logging.warning('Drive backup will be disabled.')
        args.use_drive_backup = False

    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process!!')
    c.learn()

def get_ipython():  # added this function to avoid errors if get_ipython() is not defined.
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None

if __name__ == "__main__":
    main()

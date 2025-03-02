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
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration. Games per Checkpoint
    'tempThreshold': 30,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 6000,          # Number of games moves for MCTS to simulate. 18496
    'arenaCompare': 10,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './colab/',
    'load_model': False,
    'load_folder_file': ('./05_02_25','checkpoint_1.pth.tar'),
    'starting_iteration': 1,    # Set to higher than 1 if resuming from a checkpoint
    'numItersForTrainExamplesHistory': 40,
    'verbose': True,

    # New parameters for Google Drive backup
    'use_drive_backup': True,   # Enable backup to Google Drive
    'drive_backup_path': '/content/drive/My Drive/BACKUP_FOLDER'  # Path in Google Drive to save backups
})

def main():
    # Google Drive mounting code
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_drive_backup', action='store_true')
    parser.add_argument('--drive_backup_path', type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO) #added logging basic config
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

    log.info('Starting the learning process 🎉')
    c.learn()

def get_ipython(): #added this function to avoid errors if get_ipython() is not defined.
  try:
    from IPython import get_ipython
    return get_ipython()
  except ImportError:
    return None

if __name__ == "__main__":
    main()
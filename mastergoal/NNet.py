import os
import time
import numpy as np
import sys
import logging

sys.path.append('../..')
from utils import *
from NeuralNet import NeuralNet

from .MastergoalNNet import MastergoalNNet as mnnet

args = dotdict({
    'lr': 0.01,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 64,
    'cuda': False,
    'num_channels': 512,
})

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Set up a file handler if you want to log to a file
file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.propagate = False


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.nnet = mnnet(game, args)
        self.input_shape = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        history = self.nnet.model.fit(
            x=input_boards,
            y=[target_pis, target_vs],
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=0  # Disable verbose output during training
        )

        # Log epoch information
        for epoch in range(args.epochs):
            loss = history.history['loss'][epoch]
            pi_loss = history.history['pi_loss'][epoch]
            v_loss = history.history['v_loss'][epoch]
            logger.info(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss:.4f}, Pi Loss: {pi_loss:.4f}, V Loss: {v_loss:.4f}")

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        encoded = board.encode()
        board = encoded[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board, verbose=False)

        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))

        self.nnet.model.load_weights(filepath)
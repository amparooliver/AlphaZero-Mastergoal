import os
import time
import sys
import numpy as np
import logging

from tqdm import tqdm
sys.path.append('../..')  # Add parent directory to the system path
from utils import *  # Import utility functions
from NeuralNet import NeuralNet  # Base class for neural networks

import torch
import torch.optim as optim

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

from .MastergoalNNet import MastergoalNNet as model  # Specific neural network model for Mastergoal

# Hyperparameters
args = dotdict({
    'lr': 0.01,  # Learning rate
    'momentum': 0.9,  # Momentum for SGD optimizer
    'epochs': 5,  # Number of training epochs
    'batch_size': 64,  # Batch size for training 64 normally but 128 for gpu
    'cuda': torch.cuda.is_available(),  # Check if CUDA is available for GPU usage
}) 


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        """
        Initialize the wrapper with the game and neural network model.
        Args:
            game: Game instance providing board size and action space.
        """
        self.model = model(game)  # Initialize the specific neural network
        self.input_shape = game.getBoardSize()  # Input dimensions
        self.action_size = game.getActionSize()  # Number of possible actions

        if args.cuda:
            print("Cuda AVAILABLE!")
            self.model.cuda()  # Move the model to GPU if CUDA is available


    def train(self, examples):
        """
        Train the neural network using provided examples.
        Args:
            examples: List of (board, pi, v) tuples.
        """
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)  # SGD optimizer

        for epoch in range(args.epochs):  # Train for multiple epochs
            print('EPOCH ::: ' + str(epoch + 1))
            logger.info(f'EPOCH ::: {epoch + 1}') 
            self.model.train()  # Set the model to training mode
            pi_losses = AverageMeter()  # Track policy loss
            v_losses = AverageMeter()  # Track value loss

            batch_count = int(len(examples) / args.batch_size)  # Number of batches
            t = tqdm(range(batch_count), desc='Training Net')  # Progress bar for visualization
            for _ in t:
                # Sample a batch of examples
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # Convert to PyTorch tensors
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if args.cuda:  # Move tensors to GPU if CUDA is available
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # Forward pass
                out_pi, out_v = self.model(boards)  # Predictions from the model
                l_pi = self.loss_pi(target_pis, out_pi)  # Policy loss
                l_v = self.loss_v(target_vs, out_v)  # Value loss
                total_loss = l_pi + l_v  # Total loss

                # Record the losses
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)  # Update progress bar

                # Log losses
                logger.info(f'Loss_pi: {pi_losses.avg}, Loss_v: {v_losses.avg}')
                # Backward pass and optimizer step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        Predict policy and value for a given board.
        Args:
            board: np.array representation of the board.
        Returns:
            pi: Action probabilities.
            v: Value of the board state.
        """
        start = time.time()  # Start timing

        encoded = board.encode()  # Encode the board into a neural network-compatible format
        s = torch.FloatTensor(encoded.astype(np.float64))  # Convert to tensor
        if args.cuda:
            s = s.contiguous().cuda()  # Move to GPU if necessary

        s = s.view(1, *self.input_shape)  # Add batch dimension
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            pi, v = self.model(s)  # Predict policy and value

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]  # Convert predictions to NumPy

    def loss_pi(self, targets, outputs):
        """
        Compute the policy loss (cross-entropy).
        """
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """
        Compute the value loss (mean squared error).
        """
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        Save the model's state to a file.
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({'state_dict': self.model.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        Load the model's state from a file.
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception(f"No model in path {filepath}")
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.model.load_state_dict(checkpoint['state_dict'])

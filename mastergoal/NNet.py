import os
import time
import sys
import numpy as np
import logging

from tqdm import tqdm
sys.path.append('../..')  # Add parent directory to the system path
from utils import *  # Import utility functions
from NeuralNet import NeuralNet  # Base class for neural networks
from TrainingPlotter import TrainingPlotter  # Import our new plotting class

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
    'batch_size': 256,  # Batch size for training 64 normally but 128 for gpu
    'cuda': torch.cuda.is_available(),  # Check if CUDA is available for GPU usage
    'plot_dir': 'training_plots',  # Directory to save plots
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # Device to use
}) 


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        """
        Initialize the wrapper with the game and neural network model.
        Args:
            game: Game instance providing board size and action space.
        """
        self.device = args.device
        self.model = model(game).to(self.device)  # Initialize the specific neural network and move to device
        self.input_shape = game.getBoardSize()  # Input dimensions
        self.action_size = game.getActionSize()  # Number of possible actions
        self.plotter = None  # Will be initialized during training

        if args.cuda:
            print("Cuda AVAILABLE!")


    def train(self, examples):
        """
        Train the neural network using provided examples.
        Args:
            examples: List of (board, pi, v) tuples.
        """
        # Initialize the plotter
        self.plotter = TrainingPlotter(output_dir=args.plot_dir)
        
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)  # SGD optimizer

        for epoch in range(args.epochs):  # Train for multiple epochs
            print('EPOCH ::: ' + str(epoch + 1))
            logger.info(f'EPOCH ::: {epoch + 1}') 
            self.model.train()  # Set the model to training mode
            pi_losses = AverageMeter()  # Track policy loss
            v_losses = AverageMeter()  # Track value loss

            batch_count = int(len(examples) / args.batch_size)  # Number of batches
            t = tqdm(range(batch_count), desc='Training Net')  # Progress bar for visualization
            
            # Process examples in batches to improve efficiency
            for batch_idx in t:
                # Sample a batch of examples
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # Convert to PyTorch tensors directly on the target device
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)

                # Forward pass
                out_pi, out_v = self.model(boards)  # Predictions from the model
                l_pi = self.loss_pi(target_pis, out_pi)  # Policy loss
                l_v = self.loss_v(target_vs, out_v)  # Value loss
                total_loss = l_pi + l_v  # Total loss

                # Record the losses - get CPU values only when needed for display
                pi_loss_val = l_pi.item()
                v_loss_val = l_v.item()
                pi_losses.update(pi_loss_val, boards.size(0))
                v_losses.update(v_loss_val, boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)  # Update progress bar

                # Log losses
                logger.info(f'Loss_pi: {pi_losses.avg}, Loss_v: {v_losses.avg}')
                
                # Record data for plotting
                self.plotter.record_batch(epoch + 1, batch_idx + 1, pi_loss_val, v_loss_val)
                
                # Backward pass and optimizer step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            # Save intermediate data and plots after each epoch
            self.plotter.save_data()
            self.plotter.plot_losses()
            
        # Final save of all data and plots
        self.plotter.save_data()
        self.plotter.plot_losses()
        print(f"Training data and plots saved to {self.plotter.run_dir}")
        logger.info(f"Training data and plots saved to {self.plotter.run_dir}")

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
        
        # Create tensor directly on the target device
        s = torch.FloatTensor(encoded.astype(np.float64)).to(self.device)
        
        s = s.view(1, *self.input_shape)  # Add batch dimension
        self.model.eval()  # Set the model to evaluation mode
        
        with torch.no_grad():  # Disable gradient computation
            pi, v = self.model(s)  # Predict policy and value
            # Only transfer back to CPU when returning
            pi_np = torch.exp(pi).cpu().numpy()[0]
            v_np = v.cpu().numpy()[0]

        return pi_np, v_np

    def predict_batch(self, boards_batch):
        """
        Predict policy and value for a batch of boards.
        Args:
            boards_batch: List of board states
        Returns:
            pi_batch: Batch of action probabilities
            v_batch: Batch of estimated state values
        """
        start = time.time()
        
        # Prepare encoded boards more efficiently
        encoded_boards = np.array([b.encode().astype(np.float32) for b in boards_batch])
        
        # Convert to tensors directly on the target device
        batch = torch.FloatTensor(encoded_boards).to(self.device)
        
        # Set model to evaluation mode and disable gradient computation
        self.model.eval()
        with torch.no_grad():
            pi_batch, v_batch = self.model(batch)
            # Only transfer back to CPU when returning
            pi_batch_np = torch.exp(pi_batch).cpu().numpy()
            v_batch_np = v_batch.cpu().numpy()
        
        return pi_batch_np, v_batch_np

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
        
        # Load directly to the correct device
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
import os
import shutil
import time
import random
import numpy as np
import math
import sys
import logging
sys.path.append('../..')
from utils import *
import tensorflow as tf
from tensorflow import keras
from NeuralNet import NeuralNet
import argparse
from .MastergoalNNet import MastergoalNNet as mnnet

# Verify GPU is available in Kaggle
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
print("GPU Device Name: ", tf.test.gpu_device_name())

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Set up a file handler if you want to log to a file
file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Configure console logging for Kaggle notebook
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

args = dotdict({
    'lr': 0.01,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 128,  # Increased for GPU
    'cuda': True,
    'num_channels': 512,
})

class CustomLoggingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Log the losses at the end of each epoch
        logger.info(f"Epoch {epoch + 1}/{self.params['epochs']}")
        logger.info(f"Loss: {logs.get('loss'):.5f}, Pi Loss: {logs.get('pi_loss'):.5f}, V Loss: {logs.get('v_loss'):.5f}")
        
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:  # Log every 100 batches for Kaggle
            logs = logs or {}
            logger.info(f"Batch {batch}: Loss: {logs.get('loss'):.5f}")

class NNetWrapper(NeuralNet):
    def __init__(self, game):
        # Enable mixed precision for better performance on Kaggle GPUs
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Using mixed precision for faster GPU training")
        
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
        
        # Initialize the logging callback
        logging_callback = CustomLoggingCallback()
        
        # Create a learning rate scheduler for better convergence
        def lr_scheduler(epoch, lr):
            if epoch > 3:
                return lr * 0.9  # Reduce learning rate by 10% after 3 epochs
            return lr
            
        lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
        
        # Train the model with callbacks
        history = self.nnet.model.fit(
            x=input_boards, 
            y=[target_pis, target_vs], 
            batch_size=args.batch_size, 
            epochs=args.epochs,
            callbacks=[logging_callback, lr_callback],
            validation_split=0.1  # Use 10% for validation to track overfitting
        )
        
        # Log final metrics
        logger.info(f"Training completed. Final loss: {history.history['loss'][-1]:.5f}")
        
        return history

    def predict(self, board):
        """
        board: np array with board
        """
        # preparing input
        encoded = board.encode()
        board = encoded[np.newaxis, :, :]
        
        # run
        pi, v = self.nnet.model.predict(board, verbose=False)
        return pi[0], v[0]
    
    def batch_predict(self, boards, batch_size=128):
        """
        More efficient batch prediction for Kaggle GPUs
        boards: list of board states
        """
        # Encode all boards
        encoded_boards = np.array([board.encode() for board in boards])
        
        # Process in batches to avoid memory issues on large datasets
        if len(encoded_boards) <= batch_size:
            pi_batch, v_batch = self.nnet.model.predict(encoded_boards, verbose=False)
            return pi_batch, v_batch
        
        # Process larger datasets in batches
        all_pi = []
        all_v = []
        for i in range(0, len(encoded_boards), batch_size):
            batch = encoded_boards[i:i+batch_size]
            pi_batch, v_batch = self.nnet.model.predict(batch, verbose=False)
            all_pi.append(pi_batch)
            all_v.append(v_batch)
        
        return np.concatenate(all_pi), np.concatenate(all_v)

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
        
        # For Kaggle, also save a complete model version
        full_model_path = os.path.join(folder, "full_model.h5")
        self.nnet.model.save(full_model_path)
        print(f"Full model saved to {full_model_path}")

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingPlotter:
    def __init__(self, output_dir='plots'):
        """
        Initialize the plotter with an output directory.
        
        Args:
            output_dir: Directory where plots and data will be saved
        """
        self.output_dir = output_dir
        self.training_data = {
            'epoch': [],
            'batch': [],
            'pi_loss': [],
            'v_loss': [],
            'total_loss': [],
            'timestamp': []
        }
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
            
        # Create a timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir)
        
    def record_batch(self, epoch, batch, pi_loss, v_loss):
        """
        Record metrics for a single batch.
        
        Args:
            epoch: Current epoch number
            batch: Current batch number
            pi_loss: Policy loss value
            v_loss: Value loss value
        """
        total_loss = pi_loss + v_loss
        timestamp = datetime.now()
        
        self.training_data['epoch'].append(epoch)
        self.training_data['batch'].append(batch)
        self.training_data['pi_loss'].append(pi_loss)
        self.training_data['v_loss'].append(v_loss)
        self.training_data['total_loss'].append(total_loss)
        self.training_data['timestamp'].append(timestamp)
    
    def save_data(self):
        """
        Save all recorded data to a CSV file.
        """
        df = pd.DataFrame(self.training_data)
        csv_path = os.path.join(self.run_dir, 'training_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved training data to {csv_path}")
        return csv_path
    
    def plot_losses(self):
        """
        Create and save plots of the losses over time.
        """
        if len(self.training_data['epoch']) == 0:
            print("No data to plot.")
            return
        
        # Convert data to numpy arrays for easier manipulation
        epochs = np.array(self.training_data['epoch'])
        batches = np.array(self.training_data['batch'])
        pi_losses = np.array(self.training_data['pi_loss'])
        v_losses = np.array(self.training_data['v_loss'])
        total_losses = np.array(self.training_data['total_loss'])
        
        # Create a unique batch index for the x-axis
        batch_indices = epochs * 1000 + batches
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot all losses
        plt.plot(batch_indices, pi_losses, label='Policy Loss', color='blue')
        plt.plot(batch_indices, v_losses, label='Value Loss', color='green')
        plt.plot(batch_indices, total_losses, label='Total Loss', color='red')
        
        plt.xlabel('Training Progress (Epoch * 1000 + Batch)')
        plt.ylabel('Loss')
        plt.title('Training Losses Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        loss_plot_path = os.path.join(self.run_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        print(f"Saved loss plot to {loss_plot_path}")
        
        # Create separate plots for each loss type
        self._create_individual_loss_plot(batch_indices, pi_losses, 'Policy Loss', 'blue', 'pi_loss_plot.png')
        self._create_individual_loss_plot(batch_indices, v_losses, 'Value Loss', 'green', 'v_loss_plot.png')
        self._create_individual_loss_plot(batch_indices, total_losses, 'Total Loss', 'red', 'total_loss_plot.png')
        
        # Create per-epoch average plots
        self._create_epoch_average_plot()
        
        plt.close('all')
    
    def _create_individual_loss_plot(self, batch_indices, losses, title, color, filename):
        """
        Create and save an individual loss plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(batch_indices, losses, label=title, color=color)
        plt.xlabel('Training Progress (Epoch * 1000 + Batch)')
        plt.ylabel('Loss')
        plt.title(f'{title} Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(self.run_dir, filename)
        plt.savefig(plot_path)
        print(f"Saved {title} plot to {plot_path}")
    
    def _create_epoch_average_plot(self):
        """
        Create and save plots of average losses per epoch.
        """
        df = pd.DataFrame(self.training_data)
        epoch_avg = df.groupby('epoch').mean()
        
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_avg.index, epoch_avg['pi_loss'], 'o-', label='Policy Loss', color='blue')
        plt.plot(epoch_avg.index, epoch_avg['v_loss'], 'o-', label='Value Loss', color='green')
        plt.plot(epoch_avg.index, epoch_avg['total_loss'], 'o-', label='Total Loss', color='red')
        
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Average Losses per Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(epoch_avg.index)
        
        plot_path = os.path.join(self.run_dir, 'epoch_avg_plot.png')
        plt.savefig(plot_path)
        print(f"Saved epoch average plot to {plot_path}")
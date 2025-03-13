import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MastergoalNNet(nn.Module):
    def __init__(self, game):
        super(MastergoalNNet, self).__init__()

        self.input_shape = game.getBoardSize()
        self.action_size = game.getActionSize()

        input_channels = self.input_shape[0]
        intermediate_channels = 256
        kernel_size = 3
        num_residual_blocks = 12

        # Capa de Entrada
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, intermediate_channels, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU()
        )

        # 12 bloques residuales
        self.residual_blocks = nn.Sequential()
        for i in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(intermediate_channels, intermediate_channels, kernel_size))

        # Policy Head
        self.policy_head = nn.Sequential(
            nn.Conv2d(intermediate_channels, 2, 1, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(330, self.action_size),
            nn.LogSoftmax(dim=1)
        )

        # Value Head
        self.value_head = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, 1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(165, intermediate_channels),
            nn.ReLU(),
            nn.Linear(intermediate_channels, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_blocks(x)

        pi = self.policy_head(x)
        v = self.value_head(x)

        return pi, v

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=1)
        self.bn2 =  nn.BatchNorm2d(output_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(output_channels)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

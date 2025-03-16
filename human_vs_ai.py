import Arena
from MCTS import MCTS

from mastergoal.MastergoalGame import MastergoalGame 
from mastergoal.NNet import NNetWrapper as nn
from mastergoal.MastergoalPlayers import *

import logging
import coloredlogs

import numpy as np
from utils import *

import argparse

log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

def main():

    # We define mastergoal as the game
    g = MastergoalGame()

    # Player 1 is the HUMAN (white = 1)
    humanPlayer = HumanPlayerConsole(g).play

    # Player 2 is the NeuralNet
    n1 = nn(g)
    n1.load_checkpoint(r"C:\Users\Amparo\Documents\PROYECTO-FINAL\Mastergoal-AlphaZero\14_03_putorch", "temp.pth.tar")
    # Parameters are defined for the NN
    args1 = dotdict({'numMCTSSims': 100, 'cpuct':1, 'verbose': True})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    # The interface is much easier if the human is white
    # The arena (which handles the game) is activated
    #arena = Arena.Arena(humanPlayer, n1p, g, display=(lambda x: x))
    #print("You are playing as white")
    arena = Arena.Arena(humanPlayer, n1p, g, display=(lambda x: x))
    result = arena.playGame(verbose=True)

    if(result == 1):
        print("CONGRATS! YOU WON!")
    else:
        print("Sorry, the AI won. Good luck next time!")    
if __name__ == "__main__":
    main()
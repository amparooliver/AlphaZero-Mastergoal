import Arena
from MCTS import MCTS

from mastergoal.MastergoalGame import MastergoalGame 
from mastergoal.NNet import NNetWrapper as nn
from mastergoal.MastergoalPlayers import *

import argparse

import logging
import coloredlogs

import numpy as np
from utils import *

log = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

"""
Compare model to a baseline agent that makes random moves
"""
def main():

    g = MastergoalGame()

    n1 = nn(g)
    n1.load_checkpoint(r"C:\Users\Amparo\Documents\PROYECTO-FINAL\Mastergoal-AlphaZero\14_03_putorch", "temp.pth.tar")
    args1 = dotdict({'numMCTSSims': 100, 'cpuct':1, 'verbose': True})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

    player2 = RandomPlayer(g).play

    arena = Arena.Arena(n1p, player2, g, display=(lambda x: x))

    oneWon, twoWon, draws = arena.playGames(4, verbose=True)

    print(f"BEST PTH wins:{oneWon},  losses:{twoWon},  draws:{draws}  (playing against random)")

if __name__ == "__main__":
    main()
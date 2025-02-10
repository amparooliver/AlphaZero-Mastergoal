import logging
import numpy as np
from MastergoalLogic import MastergoalBoard, Pieces

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def test_valid_moves():
    board = MastergoalBoard()

    # Set up a test scenario (empty board except for one player and ball)
    board.pieces = np.zeros((board.rows, board.cols), dtype=int)
    
    # Place a player 
    board.pieces[7, 5] = Pieces.WHITE_PLAYER
    # Place a player 
    board.pieces[9, 7] = Pieces.RED_PLAYER
    # Place a ball
    board.pieces[7, 7] = Pieces.BALL

    # Display board for verification
    print("\nTest Board State:")
    board.display()

    # Get valid moves
    valid_moves = board.getValidMoves()

    # Decode and log valid moves
    valid_moves_list = []
    for move_index in range(16):
        for kick_index in range(33):
            if valid_moves[move_index][kick_index]:
                piece_move = board.decode_move(move_index)
                ball_kick = board.decode_kick(kick_index)
                valid_moves_list.append((piece_move, ball_kick))

    log.info(f"Valid moves (decoded): {valid_moves_list}")

if __name__ == "__main__":
    test_valid_moves()

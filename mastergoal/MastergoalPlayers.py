import numpy as np
from .MastergoalGame import MastergoalGame
from .MastergoalLogic import *
import time
from queue import Empty

class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        MastergoalBoard.display(board)
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a

class HumanPlayerConsole():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)
        
        # Get current player piece position
        iPlayerR, iPlayerC = np.where(board.pieces == Pieces.WHITE_PLAYER)
        iPlayerR, iPlayerC = iPlayerR[0], iPlayerC[0]
        
        while True:
            print("-----------------------------------")
            print("Introduce fila y columna. Ej: 6 5")
            print("-----------------------------------")
            try:
                
                # Display the board in the console
                MastergoalBoard.display(board)
                input_move = input("Introduce coordenadas: ").split(" ")
                                
                nueva_f_pieza = int(input_move[0])
                nueva_c_pieza = int(input_move[1])
                
                # Validate piece move
                if not board.is_valid_move(nueva_f_pieza, nueva_c_pieza, iPlayerR, iPlayerC):
                    print("Invalid piece move. Try again.")
                    continue
                
                # Encode piece move
                piece_index = board.encode_move(iPlayerR, iPlayerC, nueva_f_pieza, nueva_c_pieza)

                # Create a copy of the board and move the piece
                board_copy = MastergoalBoard()
                board_copy.pieces = board.pieces.copy()
                board_copy.pieces[iPlayerR, iPlayerC] = Pieces.EMPTY
                board_copy.pieces[nueva_f_pieza, nueva_c_pieza] = Pieces.WHITE_PLAYER
                
                # Display board with piece moved
                board_copy.display()
                #board_copy.update_web_board()

                # Check if ball is adjacent
                if board.is_ball_adjacent(nueva_f_pieza, nueva_c_pieza):

                    # Get current ball position
                    ball_row, ball_col = board.get_ball_position()
                    
                    print("Ball is adjacent. You must kick the ball.")
                    
                    while True:  # Loop for ball input
                        try:
                            print("Introduce ball kick coordinates (new row and column):")
                            ball_input = input("New ball position: ").split(" ")
                            nueva_f_pel = int(ball_input[0])
                            nueva_c_pel = int(ball_input[1])
                            
                            # Validate ball move
                            if board.is_valid_ball_move(nueva_f_pel, nueva_c_pel, ball_row[0], ball_col[0], nueva_f_pieza, nueva_c_pieza):
                                ball_index = board.encode_kick(nueva_f_pel, nueva_c_pel, ball_row[0], ball_col[0])
                                
                                # Display board with ball moved
                                board_copy.pieces[ball_row[0], ball_col[0]] = Pieces.EMPTY
                                board_copy.pieces[nueva_f_pel, nueva_c_pel] = Pieces.BALL
                                board_copy.display()
                                
                                break  # Exit ball input loop
                            else:
                                print("Invalid ball kick. Try again.")
                        except (ValueError, IndexError):
                            print("Invalid input for ball coordinates. Try again.")
                else:
                    # No ball movement
                    ball_index = 32
                
                # Combine piece and ball move
                a = piece_index * 33 + ball_index
                
                # Final validation
                if valid[int(a)]:
                    return a
                else:
                    print("Invalid move. Try again.")
            except (ValueError, IndexError) as e:
                print(f"Invalid input. {e}")
                print("Format: piece_row piece_col new_piece_row new_piece_col [ball_row ball_col]")

class HumanPlayerWeb:
    def __init__(self, game, click_queue):
        self.game = game
        self.click_queue = click_queue

    def play(self, board):
        valid = self.game.getValidMoves(board, 1)

        # Get current player piece position
        iPlayerR, iPlayerC = np.where(board.pieces == Pieces.WHITE_PLAYER)
        iPlayerR, iPlayerC = iPlayerR[0], iPlayerC[0]

        while True:
            MastergoalBoard.display(board)
            # Update the web board state
            MastergoalBoard.update_web_board(board)
            print("Waiting for player click...")

            # Wait for a click from the queue
            try:
                nueva_f_pieza, nueva_c_pieza = self.click_queue.get(timeout=10)  # Wait up to 10 seconds
            except Empty:
                print("No click received. Retrying...")
                continue

            # Validate piece move
            if not board.is_valid_move(nueva_f_pieza, nueva_c_pieza, iPlayerR, iPlayerC):
                print("Invalid piece move. Try again.")
                continue

            # Encode piece move
            piece_index = board.encode_move(iPlayerR, iPlayerC, nueva_f_pieza, nueva_c_pieza)

            # Create a copy of the board and move the piece
            board_copy = MastergoalBoard()
            board_copy.pieces = board.pieces.copy()
            board_copy.pieces[iPlayerR, iPlayerC] = Pieces.EMPTY
            board_copy.pieces[nueva_f_pieza, nueva_c_pieza] = Pieces.WHITE_PLAYER
            
            # Display board with piece moved
            #board_copy.display()
            board_copy.update_web_board()
            board_copy.display()

            # Handle ball movement if adjacent
            if board.is_ball_adjacent(nueva_f_pieza, nueva_c_pieza):
                print("Ball is adjacent. Waiting for ball kick click...")

                valid_ball_kick = False
                while not valid_ball_kick:
                    try:
                        nueva_f_pel, nueva_c_pel = self.click_queue.get(timeout=10)  # Wait for ball kick click
                    except Empty:
                        print("No ball click received. Retrying...")
                        continue

                    ball_row, ball_col = board.get_ball_position()
                    if board.is_valid_ball_move(nueva_f_pel, nueva_c_pel, ball_row[0], ball_col[0], nueva_f_pieza, nueva_c_pieza):
                        valid_ball_kick = True
                        ball_index = board.encode_kick(nueva_f_pel, nueva_c_pel, ball_row[0], ball_col[0])     
                        # Display board with ball moved
                        board_copy.pieces[ball_row[0], ball_col[0]] = Pieces.EMPTY
                        board_copy.pieces[nueva_f_pel, nueva_c_pel] = Pieces.BALL
                        board_copy.update_web_board()
                        board_copy.display()
                    else:
                        print("Invalid ball kick. Try again.")
            else:
                ball_index = 32

            # Combine piece and ball move
            a = piece_index * 33 + ball_index
            if valid[int(a)]:
                return a
            else:
                print("Invalid move. Try again.")
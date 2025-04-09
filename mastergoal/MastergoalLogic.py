'''
Author: Amparo Oliver 
Date: September, 2024.

Board size is 15x11.
'''
import numpy as np
from enum import IntEnum
import requests

# Definición de constantes
NUM_PLANES = 5  # 1 para pelota, 1 para jugador rojo, 1 para jugador blanco, 1 para turno actual, 1 para nro de moves
ACTION_SIZE = 528  #  16 x 33

PLAYER_PIECE_LAYER = 0
OPPONENT_PIECE_LAYER = 1
BALL_LAYER = 2
PLAYER_LAYER = 3
MOVE_COUNT_LAYER = 4


class Pieces(IntEnum):
    EMPTY = 0
    RED_PLAYER = -1
    WHITE_PLAYER = 1
    BALL = 2

class MastergoalBoard():
    def __init__(self):
        self.rows = 15
        self.cols = 11
        self.pieces = self.getInitialPieces()
        self.red_turn = False
        self.red_goals = 0
        self.white_goals = 0
        self.goals_to_win = 1
        self.move_count = 0

        # Track ball
        self.ball_row = 7
        self.ball_col = 5

    def encode(self):
        # Create a zeros array with shape (5, 15, 11)
        board = np.zeros((NUM_PLANES, self.rows, self.cols))
        
        # Use boolean masks to fill in the layers
        board[PLAYER_PIECE_LAYER] = (self.pieces == Pieces.WHITE_PLAYER)
        board[OPPONENT_PIECE_LAYER] = (self.pieces == Pieces.RED_PLAYER)
        board[BALL_LAYER] = (abs(self.pieces) == Pieces.BALL)
        
        # Player turn layer (1 for white, -1 for red)
        board[PLAYER_LAYER] = 1 if not self.red_turn else -1
        
        # Move count layer
        board[MOVE_COUNT_LAYER] = self.move_count
        
        return board

    def getInitialPieces(self):
        pieces = np.zeros((self.rows, self.cols), dtype='int8')
        pieces[4, 5] = Pieces.WHITE_PLAYER
        pieces[10, 5] = Pieces.RED_PLAYER
        pieces[7, 5] = Pieces.BALL
        return pieces

    def getValidMoves(self):
        moves = np.zeros((16, 33), dtype=bool)
        player_positions = np.where(self.pieces == 1)
        for i in range(len(player_positions[0])):
            row, col = player_positions[0][i], player_positions[1][i]
            self.addPlayerMoves(moves, row, col)
        return moves

    def addPlayerMoves(self, moves, row, col):
        # Generate all potential moves in a vectorized way
        dr = np.array([-2, -2, -2, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        dc = np.array([-2, 0, 2, -1, 0, 1, -2, -1, 1, 2, -1, 0, 1, -2, 0, 2])
        
        # Filter out (0,0)
        mask = ~((dr == 0) & (dc == 0))
        dr, dc = dr[mask], dc[mask]
        
        new_rows = row + dr
        new_cols = col + dc
        
        for i in range(len(new_rows)):
            new_row, new_col = new_rows[i], new_cols[i]
            if self.is_valid_move(new_row, new_col, row, col):
                move_index = self.encode_move(row, col, new_row, new_col)
                
                if self.is_ball_adjacent(new_row, new_col):
                    self.addBallKicks(moves, move_index, new_row, new_col)
                else:
                    moves[move_index][32] = True

    def addBallKicks(self, moves, move_index, fpr, fpc):
        ball_row, ball_col = self.get_ball_position()
        
        # Create arrays for all possible kick directions
        dr_values = np.arange(-4, 5)
        dc_values = np.arange(-4, 5)
        dr, dc = np.meshgrid(dr_values, dc_values)
        dr = dr.flatten()
        dc = dc.flatten()
        
        # Remove (0,0)
        mask = ~((dr == 0) & (dc == 0))
        dr, dc = dr[mask], dc[mask]
        
        new_rows = ball_row + dr
        new_cols = ball_col + dc
        
        for i in range(len(new_rows)):
            new_row, new_col = new_rows[i], new_cols[i]
            if self.is_valid_ball_move(new_row, new_col, ball_row, ball_col, fpr, fpc):
                kick_index = self.encode_kick(new_row, new_col, ball_row, ball_col)
                moves[move_index][kick_index] = True

    def is_valid_move(self, row, col, start_row, start_col):
        # Movement Type Check
        if not self.is_diagonal_hor_ver(row, col, start_row, start_col):
            return False
        # Boundary Check
        if not (0 < row < self.rows-1 and 0 <= col < self.cols):
            return False
        # Empty Square Check
        if self.pieces[row][col] != Pieces.EMPTY:
            return False
        # Line Blocked Check
        if self.is_line_blocked(start_row, start_col, row, col):
            return False       
        # Distance Check (Just to be safe)
        row_distance = abs(row - start_row)
        col_distance = abs(col - start_col)
        if max(row_distance, col_distance) > 2:
            return False
        # Invalid Square Check
        if self.is_invalid_square(row, col):
            return False
        # Own Corner Check
        if self.is_own_corner(row, col):
            return False
        return True
  
    def is_valid_ball_move(self, row, col, start_row, start_col, fpr, fpc):
        # Boundary Check
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False  
        # Movement Type Check
        if not self.is_diagonal_hor_ver(row, col, start_row, start_col):
            return False
        # Additional existing checks
        if (not self.is_empty_space(row, col, fpr, fpc) or
            self.is_invalid_square(row, col) or
            self.is_own_area(row, col) or
            self.is_adjacent_to_player(row, col, fpr, fpc) or
            self.is_own_corner(row, col)):
            return False
        # Distance Check 4 (Just to be safe)
        row_distance = abs(row - start_row)
        col_distance = abs(col - start_col)
        if max(row_distance, col_distance) > 4:
            return False 
        return True

    def is_diagonal_hor_ver(self, row, col, start_row, start_col):
        if (
            (row == start_row) or  # Movimiento horizontal
            (col == start_col) or  # Movimiento vertical
            (abs(row - start_row) == abs(col - start_col))  # Movimiento diagonal
        ):
            return True
        return False  
    
    def is_empty_space(self,row,col,fPlayerR, fPlayerC):
        boardCopy = self.pieces.copy()
        # Encontrar la pieza del jugador inicial y eliminarla de la copia
        iPlayerR, iPlayerC = np.where(self.pieces == 1)
        boardCopy[iPlayerR, iPlayerC] = Pieces.EMPTY
        # Encontrar la posición inicial de la pelota y vaciarla en la copia
        iBallR, iBallC = self.get_ball_position()
        boardCopy[iBallR, iBallC] = Pieces.EMPTY
        # Agregar la pieza del jugador en la nueva posición a la copia
        boardCopy[fPlayerR, fPlayerC] = 1
        if (boardCopy[row][col] == Pieces.EMPTY):
            return True
        return False

    def is_invalid_square(self, row, col):
        return (row == 0 or row == 14) and (col <= 2 or col >= 8)

    def is_own_area(self, row, col):
        if (row <= 4 and 1 <= col <= 9):
            return True
        else:
            return False #row >= 10 and 1 <= col <= 9 # Since im flipping the board, the current player always has their own area on the top

    def is_adjacent_to_player(self, fBallR, fBallC, fPlayerR, fPlayerC):
        boardCopy = self.pieces.copy()
        
        # Update board copy
        iPlayerR, iPlayerC = np.where(self.pieces == 1)
        boardCopy[iPlayerR, iPlayerC] = Pieces.EMPTY
        boardCopy[fPlayerR, fPlayerC] = 1
        
        # Clear the ball position in the copy
        boardCopy[self.ball_row, self.ball_col] = Pieces.EMPTY
        
        # Generate all adjacent positions
        dr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
        dc = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
        
        adj_rows = fBallR + dr
        adj_cols = fBallC + dc
        
        # Filter positions within board boundaries
        valid_mask = (0 <= adj_rows) & (adj_rows < self.rows) & (0 <= adj_cols) & (adj_cols < self.cols)
        valid_rows = adj_rows[valid_mask]
        valid_cols = adj_cols[valid_mask]
        
        # Check if any adjacent position contains a player
        return np.any((boardCopy[valid_rows, valid_cols] == 1) | (boardCopy[valid_rows, valid_cols] == -1))

    def is_own_corner(self, row, col):
        if (row == 1 and (col == 0 or col == 10)):
            return True
        return False 

    def is_line_blocked(self, start_row, start_col, end_row, end_col):
        delta_row = end_row - start_row
        delta_col = end_col - start_col
        steps = max(abs(delta_row), abs(delta_col))
        
        if steps <= 1:
            return False
        
        # Create intermediate positions
        intermediate_rows = np.array([start_row + step * delta_row // steps for step in range(1, steps)])
        intermediate_cols = np.array([start_col + step * delta_col // steps for step in range(1, steps)])
        
        # Check if any intermediate position is occupied
        return np.any(self.pieces[intermediate_rows, intermediate_cols] != Pieces.EMPTY)
     
    def performMove(self, action, verbose):
        player_move, ball_kick = self.decode_action(action)
        start_row, start_col = np.where(self.pieces == 1)
        # Move player
        end_row, end_col = start_row + player_move[0], start_col + player_move[1]
        self.pieces[start_row, start_col] = Pieces.EMPTY
        self.pieces[end_row, end_col] = 1
        self.move_count += 1

        # Kick ball if applicable
        if ball_kick != 32:
            ball_row, ball_col = self.get_ball_position()
            new_ball_row, new_ball_col = ball_row + ball_kick[0], ball_col + ball_kick[1]
            self.pieces[ball_row, ball_col] = Pieces.EMPTY
            self.pieces[new_ball_row, new_ball_col] = Pieces.BALL
            # Update tracked ball
            self.ball_row, self.ball_col = new_ball_row, new_ball_col

        # Flipping board!!!!
        # Voltear el tablero para la perspectiva del otro jugador
        self.pieces = -1 * self.pieces
        self.pieces = np.flipud(self.pieces)  # Voltear verticalmente

        # Since we know exactly where the ball is, directly fix the ball value
        # after the flip and negation instead of searching
        if self.pieces[self.rows - 1 - self.ball_row, self.ball_col] == -Pieces.BALL:
            self.pieces[self.rows - 1 - self.ball_row, self.ball_col] = Pieces.BALL
        
        # Update ball position after flipping the board (both row and column)
        self.ball_row = self.rows - 1 - self.ball_row

        self.red_turn = not self.red_turn

    def is_ball_adjacent(self, row, col):
        return abs(row - self.ball_row) <= 1 and abs(col - self.ball_col) <= 1  

    def get_ball_position(self):
        # Return the actual scalar values, not arrays
        return self.ball_row, self.ball_col

    def is_goal(self, row):
        if (row == 14): #Current player goal
            return True
        return False

    def handle_goal(self, row):
        if self.red_turn:
            self.red_goals += 1
        else:
            self.white_goals += 1
        if self.red_goals == self.goals_to_win or self.white_goals == self.goals_to_win:
            return True
        return False

    def reset_after_goal(self):
        self.red_goals = 0
        self.white_goals = 0

    def is_game_over(self, verbose):
        if self.is_goal(self.ball_row):
            if self.red_turn:
                self.red_goals += 1
                return -1
            else:
                self.white_goals += 1
                return 1
        if (self.move_count >= 40):
            # Game taking too long, calling it a draw!
            return 1e-4     
        return 0


    def encode_move(self, start_row, start_col, end_row, end_col):
        move_vector = (end_row - start_row, end_col - start_col)
        move_map = {
            (-2, -2): 0, (-2, 0): 1, (-2, 2): 2,
            (-1, -1): 3, (-1, 0): 4, (-1, 1): 5,
            (0, -2): 6, (0, -1): 7, (0, 1): 8, (0, 2): 9,
            (1, -1): 10, (1, 0): 11, (1, 1): 12,
            (2, -2): 13, (2, 0): 14, (2, 2): 15
        }
        return move_map[move_vector]

    def encode_kick(self,end_row, end_col, start_row, start_col):
        kick_vector = (end_row - start_row, end_col - start_col)
        kick_map = {
            (-1, -1): 0, (-1, 0): 1, (-1, +1): 2, 
            (0, -1): 3,  (0, +1): 4, (+1, -1): 5,
            (+1, 0): 6, (+1, +1): 7, (-2, 0): 8, (-3, 0): 9,
            (-4, 0): 10, (+2, 0): 11, (+3, 0): 12,
            (+4, 0): 13, (0, +2): 14, (0, +3): 15,
            (0, +4): 16, (0, -2): 17, (0, -3): 18,
            (0, -4): 19, (-2, -2): 20, (-3, -3): 21,
            (-4, -4): 22, (-2, +2): 23, (-3, +3): 24,
            (-4, +4): 25, (+2, -2): 26, (+3, -3): 27,
            (+4, -4): 28, (+2, +2): 29, (+3, +3): 30,
            (+4, +4): 31
        }
        return kick_map[kick_vector]

    def decode_move(self, index):
        move_map = {
            0: (-2, -2), 1: (-2, 0), 2: (-2, +2),
            3: (-1, -1), 4: (-1, 0), 5: (-1, +1),
            6: (0, -2), 7: (0, -1), 8: (0, +1), 9: (0, +2),
            10: (+1, -1), 11: (+1, 0), 12: (+1, +1),
            13: (+2, -2), 14: (+2, 0), 15: (+2, +2)
        }
        return move_map[index]
    
    def decode_kick(self, index):
        kick_map = {
                    0: (-1, -1), 1: (-1, 0), 2: (-1, +1), 
                    3: (0, -1), 4: (0, +1), 5: (+1, -1),
                    6:(+1, 0),  7:(+1, +1), 8: (-2, 0), 9: (-3, 0),
                    10: (-4, 0), 11: (+2, 0), 12: (+3, 0),
                    13: (+4, 0), 14: (0, +2), 15: (0, +3),
                    16: (0, +4), 17: (0, -2), 18: (0, -3),
                    19: (0, -4), 20: (-2, -2), 21: (-3, -3),
                    22: (-4, -4), 23: (-2, +2), 24: (-3, +3),
                    25: (-4, +4), 26: (+2, -2), 27: (+3, -3),
                    28: (+4, -4), 29: (+2, +2), 30: (+3, +3),
                    31: (+4, +4), 32: (0,0)
                }
                
        return kick_map[index]

    def decode_action(self, indice_accion):
        """Decodifica un índice de acción en los índices de fila, columna, movimiento de pieza y movimiento de pelota."""
        num_movimientos_pieza = 16
        num_movimientos_pelota = 33

        ball_index = (indice_accion % num_movimientos_pelota)
        indice_accion //= num_movimientos_pelota
        piece_index = (indice_accion % num_movimientos_pieza) 
        piece_move = self.decode_move(piece_index) 
        ball_kick = self.decode_kick(ball_index)
        return (piece_move, ball_kick)


    def hashKey(self):
        return f"{np.array2string(self.pieces)}${self.red_turn}${self.red_goals}${self.white_goals}"

    def display(self):
        """Muestra el tablero de Mastergoal en la consola."""

        pieces = self.pieces.copy()  # Crear una copia para no modificar el original

        # Si es el turno de las blancas, invertir el tablero y las piezas para mostrarlo desde su perspectiva
        if self.red_turn:  # Asumiendo que self.red_turn indica si es el turno del jugador rojo
            pieces = -1 * pieces
            pieces = np.flipud(pieces)  # Voltear el tablero verticalmente

        # Imprimir el tablero
        print("  ", end="")
        for col in range(self.cols):
            print(col, end=" ")
        print("")
        for row in range(self.rows):
            print(row, end="  ")
            for col in range(self.cols):
                piece = pieces[row][col]
                symbol = '_'
                if piece == Pieces.RED_PLAYER: # INVERTIDOS TEMPORALMENTE
                    symbol = 'R'
                elif piece == Pieces.WHITE_PLAYER:
                    symbol = 'W'
                elif piece == Pieces.BALL or piece == -Pieces.BALL:
                    symbol = 'O'
                print(symbol, end=" ")
            print("")
        

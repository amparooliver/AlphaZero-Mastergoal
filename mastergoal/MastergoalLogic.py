'''
Author: Amparo Oliver 
Date: September, 2024.

Board size is 15x11.
'''
import numpy as np
from enum import IntEnum

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

    def encode(self):
        # Crea un array de ceros con shape (5, 15, 11)
        board = np.zeros((NUM_PLANES, self.rows, self.cols))

        # Iteramos sobre el tablero para colocar cada pieza en la capa correcta
        for r in range(self.rows):
            for c in range(self.cols):
                piece = self.pieces[r, c]
                if piece == Pieces.WHITE_PLAYER:
                    board[PLAYER_PIECE_LAYER, r, c] = 1
                elif piece == Pieces.RED_PLAYER:
                    board[OPPONENT_PIECE_LAYER, r, c] = 1
                elif abs(piece) == Pieces.BALL:
                    board[BALL_LAYER, r, c] = 1

        # Capa del jugador en turno (1 para blanco, -1 para rojo)
        if not self.red_turn:
            board[PLAYER_LAYER, :, :] = 1
        else:
            board[PLAYER_LAYER, :, :] = -1

        # Capa de conteo de movimientos (rellena todo con el número de movimientos actual)
        board[MOVE_COUNT_LAYER, :, :] = self.move_count

        return board

    def getInitialPieces(self):
        pieces = np.zeros((self.rows, self.cols), dtype='int8')
        pieces[4, 5] = Pieces.WHITE_PLAYER
        pieces[10, 5] = Pieces.RED_PLAYER
        pieces[7, 5] = Pieces.BALL
        return pieces

    def getValidMoves(self):
        moves = np.zeros((16, 33), dtype=bool)
        for row in range(self.rows):
            for col in range(self.cols):
                if self.pieces[row][col] == 1:
                    self.addPlayerMoves(moves, row, col)
        '''
        # Log readable moves
        valid_moves_list = []
        for move_index in range(16):
            for kick_index in range(33):
                if moves[move_index][kick_index]:
                    piece_move = self.decode_move(move_index)
                    ball_kick = self.decode_kick(kick_index)
                    valid_moves_list.append((piece_move, ball_kick))
        print(f"Valid moves (decoded): {valid_moves_list}")
        '''
        return moves

    def addPlayerMoves(self, moves, row, col):
        for dr in [-2, -1, 0, 1, 2]:
                    for dc in [-2, -1, 0, 1, 2]:
                        if dr == 0 and dc == 0:
                            continue
                        new_row, new_col = row + dr, col + dc
                        if self.is_valid_move(new_row, new_col, row, col):

                            move_index = self.encode_move(row, col, new_row, new_col)

                            if self.is_ball_adjacent(new_row, new_col):

                                # Player must kick the ball
                                self.addBallKicks(moves, move_index, new_row,new_col)
                            else:
                                # Player just moves without kicking
                                moves[move_index][32] = True # ADDED special key where ball does not move

    def addBallKicks(self, moves,move_index,fpr, fpc):
        ball_row, ball_col = self.get_ball_position()
        #print(f"Ball position found: {ball_row} , {ball_col}")
        for dr in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
            for dc in [-4, -3, -2, -1, 0, 1, 2, 3, 4]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = ball_row[0] + dr, ball_col[0] + dc
                if self.is_valid_ball_move(new_row, new_col, ball_row[0], ball_col[0],fpr, fpc):
                    kick_index = self.encode_kick(new_row, new_col, ball_row[0], ball_col[0])
                    moves[move_index][kick_index] = True

    def is_valid_move(self, row, col, start_row, start_col):
        # Check if the move is within board boundaries
        if not (0 < row < self.rows-1 and 0 <= col < self.cols):
            return False
        
        # Check if the destination square is empty
        if self.pieces[row][col] != Pieces.EMPTY:
            return False
        
        # Check for invalid square
        if self.is_invalid_square(row, col):
            return False
        
        #Check for own corner
        if self.is_own_corner(row, col):
            return False
        
        #Check if the move jumps above ball or player
        if self.is_line_blocked(start_row, start_col, row, col):
            return False
        
        # Check diagonal, horizontal, or vertical movement
        if not self.is_diagonal_hor_ver(row, col, start_row, start_col):
            return False
        
        # Check distance restriction (up to 2 squares)
        row_distance = abs(row - start_row)
        col_distance = abs(col - start_col)
        
        # Allow moves that are within 2 squares in any direction
        # This includes diagonal moves where row and column distances match
        if max(row_distance, col_distance) > 2:
            return False
        
        return True
  
    def is_valid_ball_move(self, row, col, start_row, start_col, fpr, fpc):
        # Check if the move is within board boundaries
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        
        # Check diagonal, horizontal, or vertical movement
        if not self.is_diagonal_hor_ver(row, col, start_row, start_col):
            return False
        
        # Check distance restriction (up to 4 squares)
        row_distance = abs(row - start_row)
        col_distance = abs(col - start_col)
        
        # Allow moves that are within 4 squares in any direction
        if max(row_distance, col_distance) > 4:
            return False
        
        # Additional existing checks
        if (not self.is_empty_space(row, col, fpr, fpc) or
            self.is_invalid_square(row, col) or
            self.is_own_area(row, col) or
            self.is_adjacent_to_player(row, col, fpr, fpc) or
            self.is_own_corner(row, col)):
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

        # Encontrar la pieza del jugador inicial y eliminarla de la copia
        iPlayerR, iPlayerC = np.where(self.pieces == 1)
        boardCopy[iPlayerR, iPlayerC] = Pieces.EMPTY

        # Agregar la pieza del jugador en la nueva posición a la copia
        boardCopy[fPlayerR, fPlayerC] = 1

        # Encontrar la posición inicial de la pelota y vaciarla en la copia
        iBallR, iBallC = self.get_ball_position()
        boardCopy[iBallR, iBallC] = Pieces.EMPTY
        # Verificar si la nueva posición de la pelota es adyacente a un jugador en la copia del tablero
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row = fBallR + dr
                new_col = fBallC + dc
                #print(f"PELOTA. new_row = {new_row} && new_col = {new_col}")
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols and \
                    (boardCopy[new_row, new_col] == 1 or boardCopy[new_row, new_col] == -1 ):
                    return True
        return False

    def is_own_corner(self, row, col):
        if (row == 1 and (col == 0 or col == 10)):
            return True
        else:
            return False # (row == 13 and (col == 0 or col == 10)) # Same logic to area

    def is_line_blocked(self, start_row, start_col, end_row, end_col):
        # Verifica si la línea pasa por encima de la pelota o de otro jugador
        delta_row = end_row - start_row
        delta_col = end_col - start_col
        steps = max(abs(delta_row), abs(delta_col))
        for step in range(1, steps):
            intermediate_row = start_row + step * delta_row // steps
            intermediate_col = start_col + step * delta_col // steps
            if self.pieces[intermediate_row][intermediate_col] in [Pieces.RED_PLAYER, Pieces.WHITE_PLAYER, Pieces.BALL]:
                return True
        return False 
     
    def performMove(self, action, verbose):

        player_move, ball_kick = self.decode_action(action)
        #print(f"player_move {player_move} y ball_kick: {ball_kick}")
        start_row, start_col = np.where(self.pieces == 1)
        # Move player
        end_row, end_col = start_row + player_move[0], start_col + player_move[1]
        self.pieces[start_row, start_col] = Pieces.EMPTY
        self.pieces[end_row, end_col] = 1
        self.move_count += 1
        #print(f"Move count is: {self.move_count}")

        # Kick ball if applicable
        if ball_kick != 32:
            ball_row, ball_col = self.get_ball_position()
            new_ball_row, new_ball_col = ball_row + ball_kick[0], ball_col + ball_kick[1]
            self.pieces[ball_row, ball_col] = Pieces.EMPTY
            self.pieces[new_ball_row, new_ball_col] = Pieces.BALL

        # Flipping board!!!!
        # Voltear el tablero para la perspectiva del otro jugador

        self.pieces = -1 * self.pieces
        self.pieces = np.flipud(self.pieces)  # Voltear verticalmente

        # Find any -2 values and convert them to 2
        ball_neg_pos = np.where(self.pieces == -Pieces.BALL)
        if len(ball_neg_pos[0]) > 0:
            self.pieces[ball_neg_pos[0], ball_neg_pos[1]] = Pieces.BALL

        self.red_turn = not self.red_turn

    def is_ball_adjacent(self, row, col):
        ball_row, ball_col = self.get_ball_position()
        return abs(row - ball_row) <= 1 and abs(col - ball_col) <= 1

    def get_ball_position(self):
        return np.where((self.pieces == Pieces.BALL) | (self.pieces == -Pieces.BALL))

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
        ball_row, ball_col = self.get_ball_position()
        if self.is_goal(ball_row):
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
        #print(f"Acaba de jugar: {'Blanco' if self.red_turn else 'Rojo'}")  # Indicar el turno del jugador
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
        
        #print(f"Ahora es el turno del jugador: {'Rojo' if self.red_turn else 'Blanco'}")  # Indicar el turno del jugador

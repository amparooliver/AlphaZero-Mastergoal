import logging
import math
import numpy as np

from collections import defaultdict

EPS = 1e-8

log = logging.getLogger(__name__)

# A dedicated logger for MCTS debugging
debug_log = logging.getLogger('MCTS_Debug')
debug_log.setLevel(logging.DEBUG)
debug_log.propagate = False  # Prevent propagation to other loggers

# Configuration
file_handler = logging.FileHandler('mcts_debug2.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
debug_log.addHandler(file_handler)

class TreeLevel():
    """
    Holds all the nodes at a certain tree depth.
    This structure helps in pruning higher levels as the game progresses.
    """
    def __init__(self):
        self.Qsa = {}  # Stores Q-values for state-action pairs (s, a)
        self.Nsa = {}  # Counts how many times edge (s, a) was visited
        self.Ns = {}  # Counts how many times state s was visited
        self.Ps = {}  # Stores the initial policy (action probabilities) from the neural network
        self.Es = {}  # Caches game-end status for state s
        self.Vs = {}  # Caches valid moves for state s

class MCTS():
    """
    Handles the Monte Carlo Tree Search (MCTS) process.
    """
    def __init__(self, game, nnet, args):
        self.game = game  # Game object providing game rules and state transitions
        self.nnet = nnet  # Neural network predicting policy and value for states
        self.args = args  # Arguments containing hyperparameters like cpuct and numMCTSSims
        self.nodes = defaultdict(TreeLevel)  # Tree structure storing information for each depth

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Executes multiple MCTS simulations from the given board state.
        
        Args:
            canonicalBoard: The current game state in its canonical form.
            temp: Temperature parameter controlling exploration (high temp = more exploration).

        Returns:
            probs: A vector where each entry represents the probability of selecting an action.
        """
        # Perform MCTS simulations
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)
            #debug_log.debug("SIMULATION DONE")

        # Serialize the board state
        s = self.game.stringRepresentation(canonicalBoard)  
        # Track the current depth of the tree
        depth = canonicalBoard.move_count  

        # Retrieve visit counts for each action
        counts = [self.nodes[depth].Nsa[(s, a)] if (s, a) in self.nodes[depth].Nsa else 0 for a in range(self.game.getActionSize())]
        #debug_log.debug(f"Counts: {counts}")

        # Discard the previous depth's nodes to save memory
        if (depth - 1) in self.nodes:
            del self.nodes[depth - 1]

        # Return a deterministic policy if temp == 0 (select most visited action)
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            #debug_log.debug(f"PROBS BEFORE: {probs}")
            probs[bestA] = 1
            #debug_log.debug(f"PROBS AFTER: {probs}")
            return probs

        # Compute a probabilistic policy based on visit counts
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]

        #debug_log.debug(f"PROBS NON DETERMINISTIC: {probs}")
        return probs

    def search(self, canonicalBoard):
        """
        Performs one iteration of MCTS, recursively exploring the tree until a leaf node is found.

        Args:
            canonicalBoard: The current game state in canonical form.

        Returns:
            v: The negative value of the board state as evaluated by the neural network.
        """
        s = self.game.stringRepresentation(canonicalBoard)  # Serialize the board state
        depth = canonicalBoard.move_count  # Track tree depth based on move count

        # Log state information
        #debug_log.debug(f"A. State:  \n{s}") 
        #debug_log.debug(f"B. Depth: {depth}")
        # Check if the game has ended for this state
        if s not in self.nodes[depth].Es:
            self.nodes[depth].Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.nodes[depth].Es[s] != 0:
            # Return the game's result if it's a terminal state
            #debug_log.debug(f"Terminal State: {self.nodes[depth].Es[s]}")
            return -self.nodes[depth].Es[s]

        # Expand the tree at a leaf node
        if s not in self.nodes[depth].Ps:
            # Query the neural network for policy (P) and value (v)
            self.nodes[depth].Ps[s], v = self.nnet.predict(canonicalBoard)
            #debug_log.debug(f"C. Policy: {self.nodes[depth].Ps[s]}, Value: {v}")
            valids = self.game.getValidMoves(canonicalBoard, 1)  # Get valid moves for the state
            self.nodes[depth].Ps[s] = self.nodes[depth].Ps[s] * valids  # Mask invalid moves

            sum_Ps_s = np.sum(self.nodes[depth].Ps[s])
            if sum_Ps_s > 0:
                self.nodes[depth].Ps[s] /= sum_Ps_s  # Normalize the policy
            else:
                # Handle edge case where no valid moves are available
                log.error("All valid moves were masked, normalizing equally.")
                print(valids)
                self.nodes[depth].Ps[s] = self.nodes[depth].Ps[s] + valids
                self.nodes[depth].Ps[s] /= np.sum(self.nodes[depth].Ps[s])

            self.nodes[depth].Vs[s] = valids  # Cache valid moves
            self.nodes[depth].Ns[s] = 0  # Initialize visit count for the state
            return -v

        # Select the action with the highest Upper Confidence Bound (UCB)

        valids = self.nodes[depth].Vs[s]  # Retrieve valid moves
        cur_best = -float('inf')  # Track the best UCB value
        best_act = -1  # Track the best action

        for a in range(self.game.getActionSize()):
            if valids[a]:

                if (s, a) in self.nodes[depth].Qsa:
                    # Use the Q-value and exploration term for visited actions
                    u = self.nodes[depth].Qsa[(s, a)] + self.args.cpuct * self.nodes[depth].Ps[s][a] * math.sqrt(self.nodes[depth].Ns[s]) / (
                            1 + self.nodes[depth].Nsa[(s, a)])
                else:
                    # Exploration term for unvisited actions (Q = 0 initially)
                    u = self.args.cpuct * self.nodes[depth].Ps[s][a] * math.sqrt(self.nodes[depth].Ns[s] + EPS)

                if u > cur_best:
                    cur_best = u
                    best_act = a

                # Log UCB values
                #debug_log.debug(f"D. Action: {a}, UCB: {u}, Q-value: {self.nodes[depth].Qsa.get((s, a), 0)}, Nsa: {self.nodes[depth].Nsa.get((s, a), 0)}")

        # Log best action
        #debug_log.debug(f"Best Action: {best_act}, UCB: {cur_best}")

        # Execute the chosen action and recurse
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)  # Get next state and player
        next_s = self.game.getCanonicalForm(next_s, next_player)  # Convert to canonical form

        # Recur on the next state and get its value
        v = self.search(next_s)

        # Update Qsa, Nsa, and Ns for the current state-action pair
        if (s, a) in self.nodes[depth].Qsa:
            self.nodes[depth].Qsa[(s, a)] = (self.nodes[depth].Nsa[(s, a)] * self.nodes[depth].Qsa[(s, a)] + v) / (self.nodes[depth].Nsa[(s, a)] + 1)
            #debug_log.debug(f"a: {a}, updated Q vale formula: ( {self.nodes[depth].Nsa[(s, a)]} * {self.nodes[depth].Qsa[(s, a)] + v} ) / {self.nodes[depth].Nsa[(s, a)]} + 1")

            self.nodes[depth].Nsa[(s, a)] += 1
        else:
            self.nodes[depth].Qsa[(s, a)] = v
            self.nodes[depth].Nsa[(s, a)] = 1
    
        self.nodes[depth].Ns[s] += 1  # Increment visit count for the state
        return -v

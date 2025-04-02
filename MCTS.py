import logging
import math
import numpy as np

from collections import defaultdict

EPS = 1e-8

log = logging.getLogger(__name__)

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

        # Serialize the board state
        s = self.game.stringRepresentation(canonicalBoard)  
        # Track the current depth of the tree
        depth = canonicalBoard.move_count  
        
        # Get action size once
        action_size = self.game.getActionSize()
        
        # Create a NumPy array to store counts for vectorization
        counts = np.zeros(action_size, dtype=np.float32)
        
        # Use vectorized operations where possible
        for a in range(action_size):
            if (s, a) in self.nodes[depth].Nsa:
                counts[a] = self.nodes[depth].Nsa[(s, a)]
        
        # Discard the previous depth's nodes to save memory
        if (depth - 1) in self.nodes:
            del self.nodes[depth - 1]

        # Return a deterministic policy if temp == 0 (select most visited action)
        if temp == 0:
            max_count = np.max(counts)
            if max_count > 0:
                # Use NumPy to find indices of maximum values
                bestAs = np.where(counts == max_count)[0]
                bestA = np.random.choice(bestAs)
                probs = np.zeros(action_size)
                probs[bestA] = 1
                return probs
            else:
                # If no moves were visited, return uniform probabilities
                return np.ones(action_size) / action_size

        # Compute a probabilistic policy based on visit counts
        if temp != 1:
            counts = np.power(counts, 1.0 / temp)
        
        # Handle the case when all counts are zero
        counts_sum = np.sum(counts)
        if counts_sum > 0:
            probs = counts / counts_sum
        else:
            # Uniform distribution if no visits
            probs = np.ones(action_size) / action_size

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

        # Check if the game has ended for this state
        if s not in self.nodes[depth].Es:
            self.nodes[depth].Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.nodes[depth].Es[s] != 0:
            # Return the game's result if it's a terminal state
            return -self.nodes[depth].Es[s]

        # Expand the tree at a leaf node
        if s not in self.nodes[depth].Ps:
            # Query the neural network for policy (P) and value (v)
            self.nodes[depth].Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)  # Get valid moves for the state
            
            # Vectorized masking of invalid moves
            self.nodes[depth].Ps[s] = self.nodes[depth].Ps[s] * valids  # Mask invalid moves

            sum_Ps_s = np.sum(self.nodes[depth].Ps[s])
            if sum_Ps_s > 0:
                self.nodes[depth].Ps[s] /= sum_Ps_s  # Normalize the policy
            else:
                log.error("All valid moves were masked, normalizing equally.")
                self.nodes[depth].Ps[s] = self.nodes[depth].Ps[s] + valids
                self.nodes[depth].Ps[s] /= np.sum(self.nodes[depth].Ps[s])

            self.nodes[depth].Vs[s] = valids  # Cache valid moves
            self.nodes[depth].Ns[s] = 0  # Initialize visit count for the state
            return -v

        # Vectorized UCB calculation
        valids = self.nodes[depth].Vs[s]  # Retrieve valid moves
        action_size = self.game.getActionSize()
        
        # Create arrays for UCB calculation
        ucb_values = np.zeros(action_size)
        
        # The policy values for all actions at state s
        policy = self.nodes[depth].Ps[s]
        
        # The exploration term constant
        cpuct = self.args.cpuct
        
        # Calculate the exploration base (sqrt of total visits)
        sqrt_total_visits = math.sqrt(self.nodes[depth].Ns[s])
        
        for a in range(action_size):
            if valids[a]:
                if (s, a) in self.nodes[depth].Qsa:
                    # Exploitation + exploration for visited nodes
                    q_value = self.nodes[depth].Qsa[(s, a)]
                    n_visits = self.nodes[depth].Nsa[(s, a)]
                    ucb_values[a] = q_value + cpuct * policy[a] * sqrt_total_visits / (1 + n_visits)
                else:
                    # Pure exploration for unvisited nodes
                    ucb_values[a] = cpuct * policy[a] * sqrt_total_visits
            else:
                # Invalid moves get -infinity
                ucb_values[a] = -float('inf')
                
        # Find the best action (argmax)
        best_act = np.argmax(ucb_values)

        # Execute the chosen action and recurse
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)  # Get next state and player
        next_s = self.game.getCanonicalForm(next_s, next_player)  # Convert to canonical form

        # Recur on the next state and get its value
        v = self.search(next_s)

        # Update Qsa, Nsa, and Ns for the current state-action pair
        if (s, a) in self.nodes[depth].Qsa:
            # Vectorized update rule
            n_visits = self.nodes[depth].Nsa[(s, a)]
            q_value = self.nodes[depth].Qsa[(s, a)]
            self.nodes[depth].Qsa[(s, a)] = (n_visits * q_value + v) / (n_visits + 1)   
            self.nodes[depth].Nsa[(s, a)] += 1
        else:
            self.nodes[depth].Qsa[(s, a)] = v
            self.nodes[depth].Nsa[(s, a)] = 1
    
        self.nodes[depth].Ns[s] += 1  # Increment visit count for the state
        return -v
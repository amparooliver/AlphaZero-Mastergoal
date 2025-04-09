import logging
import math
import numpy as np
import torch
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
        self.Ns = {}   # Counts how many times state s was visited
        self.Ps = {}   # Stores the initial policy (action probabilities) from the neural network
        self.Es = {}   # Caches game-end status for state s
        self.Vs = {}   # Caches valid moves for state s

class MCTS():
    """
    Modified MCTS with batch evaluation of leaf nodes.
    Adjusted to continue searches interrupted by batch processing.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.nodes = defaultdict(TreeLevel)

        # Batch processing parameters
        # Consider tuning this based on your GPU memory and desired latency
        self.batch_size = getattr(args, 'mctsBatchSize', 128) # Use batch size from args if available

        # Structures to track search paths for batch evaluation
        self.pending_searches = []  # List of search paths waiting for evaluation
        self.leaf_nodes = []        # List of leaf node board objects for batch evaluation
        self.leaf_states = []       # Corresponding state string representations
        self.leaf_depths = []       # Corresponding tree depths

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Performs multiple MCTS simulations with batch evaluation of leaf nodes.
        Returns action probabilities.
        """
        # Clear any previous batch data before starting new action selection
        self._clear_batch_data()

        # Execute MCTS simulations
        for i in range(self.args.numMCTSSims):
            # Start a new search from the root
            self.search(canonicalBoard) # The search function now handles batch processing internally

        # Process any remaining leaf nodes at the end of all simulations
        if len(self.leaf_nodes) > 0:
            self._process_batch_and_backpropagate() # Process the final batch

        # --- Rest of the getActionProb function remains the same ---

        # Get the state representation and depth
        s = self.game.stringRepresentation(canonicalBoard)
        depth = canonicalBoard.move_count # Assuming board object tracks move count/depth
        action_size = self.game.getActionSize()

        # Create array for visit counts
        counts = np.zeros(action_size, dtype=np.float32)

        # Populate visit counts from the root node
        current_level = self.nodes[depth]
        if s in current_level.Ns: # Check if root was visited
            for a in range(action_size):
                if (s, a) in current_level.Nsa:
                    counts[a] = current_level.Nsa[(s, a)]
        else:
             # Should not happen if numMCTSSims > 0 and root is not terminal, but handle defensively
            log.warning(f"Root node {s} at depth {depth} not found in MCTS statistics after simulations.")
            # Fallback: use uniform probability or prior policy if available
            valid_moves = self.game.getValidMoves(canonicalBoard, 1)
            num_valid = np.sum(valid_moves)
            if num_valid > 0:
                return valid_moves / num_valid
            else: # Should be terminal state if no valid moves
                 return np.zeros(action_size) # Or handle terminal case appropriately


        # Clean up previous depth to save memory (optional, consider if memory is tight)
        # Be careful if using this during training phases where state might be revisited
        # if (depth - 1) in self.nodes:
        #    del self.nodes[depth - 1]

        # Handle temperature parameter
        if temp == 0:  # Deterministic policy: Choose the most visited action
            max_count = np.max(counts)
            # Ensure there's at least one visit before selecting best action
            if max_count > 0:
                 # Break ties randomly among best actions
                bestAs = np.where(counts == max_count)[0]
                bestA = np.random.choice(bestAs)
                probs = np.zeros(action_size)
                probs[bestA] = 1
                return probs
            else:
                # If no visits (e.g., only 1 sim and it hit batch limit immediately), return uniform over valid
                log.warning("No visits recorded for root node actions, returning uniform over valid moves.")
                valid_moves = self.game.getValidMoves(canonicalBoard, 1)
                num_valid = np.sum(valid_moves)
                if num_valid > 0:
                    return valid_moves / num_valid
                else: # Should be terminal state if no valid moves
                    return np.zeros(action_size)


        # Apply temperature scaling to visit counts
        if temp != 1:
             # Prevent issues with temp=0 or negative counts (though counts should be >= 0)
            counts = np.power(np.maximum(counts, 0), 1.0 / temp)

        # Normalize to get probabilities
        counts_sum = np.sum(counts)
        if counts_sum > 0:
            probs = counts / counts_sum
        else:
            # Fallback if sum is zero (e.g., after temp scaling of small counts)
            log.warning("Sum of counts is zero after temperature application, returning uniform over valid moves.")
            valid_moves = self.game.getValidMoves(canonicalBoard, 1)
            num_valid = np.sum(valid_moves)
            if num_valid > 0:
                probs = valid_moves / num_valid
            else: # Should be terminal state if no valid moves
                probs = np.zeros(action_size)


        return probs

    def search(self, canonicalBoard, search_path=None):
        """
        Performs one MCTS search iteration with support for batch evaluation.
        If a batch becomes full during the search, it processes the batch
        and *continues* the current search path from the now-expanded node.

        Args:
            canonicalBoard: Current board state (must be canonical form).
            search_path: List of (state_str, depth, action) tuples tracing the path.

        Returns:
            v: The negated value backup for the parent node. Returns 0 for terminal states
               visited for the first time in this path, or the actual backed-up value.
               Returns the value directly if the path was completed without pending batch.
        """
        # Initialize search path if this is the root call
        if search_path is None:
            search_path = []

        s = self.game.stringRepresentation(canonicalBoard)
        # Assuming the board object has a way to track its depth/move count
        depth = canonicalBoard.move_count
        current_level = self.nodes[depth] # Get the dictionary for the current depth


        # 1. Check if terminal state
        if s not in current_level.Es:
            current_level.Es[s] = self.game.getGameEnded(canonicalBoard, 1) # Player 1 perspective

        if current_level.Es[s] != 0:
            # Terminal state reached. Return the game result (negated for the parent).
            # The value is from the perspective of the *current* player (player 1 in canonical form).
            # The parent node made the move leading here, so it receives the negative value.
            return -current_level.Es[s]

        # 2. Check if leaf node (not yet expanded by NN)
        if s not in current_level.Ps:
            # --- This is a leaf node ---
            # Add to batch for neural network evaluation
            self.leaf_nodes.append(canonicalBoard)
            self.leaf_states.append(s)
            self.leaf_depths.append(depth)
            # Add the final step (leaf state, depth, no action yet) to this path
            current_path_with_leaf = search_path + [(s, depth, None)]
            self.pending_searches.append(current_path_with_leaf)

            # Check if batch is full
            if len(self.leaf_nodes) >= self.batch_size:
                # Process the full batch *now*
                self._process_batch_and_backpropagate()
                # --- CRITICAL CHANGE ---
                # The batch processing *expanded* this node (s).
                # We don't return None anymore. The search continues below
                # as if we just arrived at an *already expanded* node.
                # The value 'v' needed for backpropagation up the *current*
                # recursive call stack will be determined by the UCB selection
                # and subsequent recursive call below.
                pass # Explicitly do nothing here and let execution continue
            else:
                # Batch is not full, this search path must wait.
                # Return None up the stack to signal that the value is pending.
                # The backpropagation for *this path* will happen later
                # when _process_batch_and_backpropagate is eventually called.
                return None # Signal pending evaluation

        # --- If we reach here, the node 's' is already expanded (either previously or by batch processing just now) ---

        # 3. Select the best action using PUCT/UCB
        if s not in current_level.Vs: # Ensure valid moves are cached
             current_level.Vs[s] = self.game.getValidMoves(canonicalBoard, 1)
        valids = current_level.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # Iterate through valid actions to find the best one according to UCB
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in current_level.Qsa:
                    # Standard UCB formula for visited actions
                    q_value = current_level.Qsa[(s, a)]
                    n_sa = current_level.Nsa[(s, a)]
                    u = q_value + self.args.cpuct * current_level.Ps[s][a] * \
                        math.sqrt(current_level.Ns[s]) / (1 + n_sa)
                else:
                    # Initialize UCB for unvisited actions (relying on prior probability)
                    # Ns[s] might be 0 if this node was *just* expanded by batch processing
                    u = self.args.cpuct * current_level.Ps[s][a] * \
                        math.sqrt(current_level.Ns[s] + EPS) # Add EPS to avoid sqrt(0)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        # Check if a valid action was found
        if best_act == -1:
            # This might happen if Ps[s] assigned zero probability to all valid moves
            # Or if there are no valid moves (which should have been caught by getGameEnded)
            log.error(f"MCTS: No valid action chosen for state {s} at depth {depth}. Valid moves: {valids}. Ps[s]: {current_level.Ps[s] if s in current_level.Ps else 'Not Expanded'}")
            # As a fallback, maybe return 0 or raise an error
            # Returning 0 might unbalance the tree if it happens often.
            # Consider if choosing any valid move randomly is better.
            valid_indices = np.where(valids == 1)[0]
            if len(valid_indices) > 0:
                 log.warning("Choosing a random valid action as fallback.")
                 best_act = np.random.choice(valid_indices)
            else:
                 log.error("No valid moves available, but not detected as terminal state.")
                 return 0 # Default return value if truly stuck


        a = best_act

        # 4. Recurse down the chosen action
        next_board, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_canonicalBoard = self.game.getCanonicalForm(next_board, next_player)

        # Append current step (state, depth, action taken) to the path for the recursive call
        path_element = (s, depth, a)
        v = self.search(next_canonicalBoard, search_path + [path_element])

        # 5. Backpropagate the result `v` if the recursive call completed
        if v is not None:
            # The recursive call returned a value (either terminal or completed path)
            # Update Q-value and visit counts for the action taken (s, a)
            if (s, a) in current_level.Qsa:
                current_level.Qsa[(s, a)] = (current_level.Nsa[(s, a)] * current_level.Qsa[(s, a)] + v) / \
                                           (current_level.Nsa[(s, a)] + 1)
                current_level.Nsa[(s, a)] += 1
            else:
                current_level.Qsa[(s, a)] = v
                current_level.Nsa[(s, a)] = 1

            # Update visit count for the state s
            current_level.Ns[s] += 1
            # Return the negated value for the parent node in the recursion
            return -v
        else:
            # The recursive call returned None, meaning it hit a leaf node and is pending batch evaluation.
            # We also return None up the stack, as this path's value is not yet determined.
            # The statistics for (s,a) and s will be updated later when the batch containing
            # the downstream leaf node is processed by _process_batch_and_backpropagate.
            return None


    def _process_batch_and_backpropagate(self):
        """
        Process all leaf nodes currently in the batch using the neural network
        and backpropagate the evaluated values 'v' up their respective search paths.
        """
        if not self.leaf_nodes:
            log.debug("Process batch called with no leaf nodes.")
            return # Nothing to process

        # Make copies of current batch data, as it will be cleared
        current_leaf_boards = self.leaf_nodes
        current_leaf_states = self.leaf_states
        current_leaf_depths = self.leaf_depths
        current_pending_searches = self.pending_searches

        # Clear global batch lists immediately to allow new searches to add nodes
        self._clear_batch_data()

        # Prepare batch for neural network
        # Assuming board object has an `encode` method for NN input
        try:
             batch_encoded = [board.encode() for board in current_leaf_boards]
             batch_tensor = torch.FloatTensor(np.array(batch_encoded).astype(np.float32))
        except Exception as e:
             log.error(f"Error encoding batch boards: {e}")
             # Handle error: maybe skip batch, log details, etc.
             # For now, just return to avoid crashing
             return


        # Move tensor to GPU if configured
        if torch.cuda.is_available() and self.args.cuda:
            batch_tensor = batch_tensor.contiguous().cuda(non_blocking=True)

        # Perform batch inference
        self.nnet.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            try:
                log_batch_pi, batch_v = self.nnet.model(batch_tensor)
                batch_pi = torch.exp(log_batch_pi).cpu().numpy() # Convert log probs to probs
                batch_v = batch_v.cpu().numpy()
            except Exception as e:
                log.error(f"Error during NN inference: {e}")
                # Handle error: maybe assign default values? For now, return.
                return


        # --- Update MCTS tree nodes with NN results ---
        for i in range(len(current_leaf_boards)):
            board = current_leaf_boards[i]
            s = current_leaf_states[i]
            depth = current_leaf_depths[i]
            pi, v = batch_pi[i], batch_v[i][0] # Extract policy and value for this node

            current_level = self.nodes[depth]

            # Ensure state exists in Es (should have been checked in search)
            if s not in current_level.Es:
                 current_level.Es[s] = self.game.getGameEnded(board, 1)
            if current_level.Es[s] != 0:
                 # If the game ended *while waiting* for batch processing (e.g. timeout?), use game result
                 log.warning(f"Node {s} at depth {depth} became terminal while pending NN eval. Using game result.")
                 v = current_level.Es[s] # Use actual game outcome instead of NN prediction
                 # Don't store policy for terminal nodes
                 current_level.Ps[s] = np.zeros(self.game.getActionSize())
                 current_level.Vs[s] = np.zeros(self.game.getActionSize()) # No valid moves
            else:
                # Store policy prediction from NN
                if s not in current_level.Vs: # Get valid moves if not already cached
                    current_level.Vs[s] = self.game.getValidMoves(board, 1)
                valids = current_level.Vs[s]
                masked_pi = pi * valids # Mask invalid actions in the policy

                sum_ps = np.sum(masked_pi)
                if sum_ps > 0:
                    current_level.Ps[s] = masked_pi / sum_ps # Normalize masked policy
                else:
                    # If NN assigns 0 probability to all valid moves, or no valid moves exist (should be terminal)
                    log.warning(f"NN predicted zero probability for all valid moves for state {s} at depth {depth}. Ps: {pi}, Valids: {valids}. Using uniform.")
                    # Fallback: Assign uniform probability to valid moves
                    num_valid = np.sum(valids)
                    if num_valid > 0:
                        current_level.Ps[s] = valids / num_valid
                    else:
                         # Should have been caught by Es check, but handle defensively
                        log.error("No valid moves but not marked terminal during batch processing.")
                        current_level.Ps[s] = np.zeros(self.game.getActionSize())


            # Initialize visit count for the newly expanded node
            current_level.Ns[s] = 0 # Initialize state visit count (will be incremented during backprop)

        # --- Backpropagate values through completed search paths ---
        for i in range(len(current_pending_searches)):
            path = current_pending_searches[i]
            # Get the NN's value prediction for the leaf node of this path
            # Check if the node became terminal while waiting
            leaf_s = current_leaf_states[i]
            leaf_depth = current_leaf_depths[i]
            leaf_level = self.nodes[leaf_depth]
            if leaf_s in leaf_level.Es and leaf_level.Es[leaf_s] != 0:
                 v = -leaf_level.Es[leaf_s] # Use the (negated) game outcome
            else:
                 v = -batch_v[i][0] # Use the (negated) NN value prediction
                                   # Negate because the value 'v' is for the *current* player at the leaf,
                                   # but we backpropagate for the *parent* player.

            # Iterate backwards through the path (excluding the leaf node itself, which has no action 'a')
            # The leaf node's (s, depth, None) entry is handled by initializing 'v' above.
            for s_path, depth_path, a_path in reversed(path[:-1]):
                 path_level = self.nodes[depth_path]
                 # Update statistics for the edge (s_path, a_path)
                 if (s_path, a_path) in path_level.Qsa:
                     path_level.Qsa[(s_path, a_path)] = (path_level.Nsa[(s_path, a_path)] * path_level.Qsa[(s_path, a_path)] + v) / \
                                                        (path_level.Nsa[(s_path, a_path)] + 1)
                     path_level.Nsa[(s_path, a_path)] += 1
                 else:
                     # This might happen if the path was interrupted very early? Should be rare.
                     log.warning(f"Backpropagating to uninitialized Qsa for ({s_path}, {a_path}) at depth {depth_path}. Initializing.")
                     path_level.Qsa[(s_path, a_path)] = v
                     path_level.Nsa[(s_path, a_path)] = 1

                 # Update visit count for the state s_path
                 # Check if Ns exists first (it should if Qsa was present)
                 if s_path not in path_level.Ns:
                     log.warning(f"Backpropagating to uninitialized Ns for state {s_path} at depth {depth_path}. Initializing.")
                     path_level.Ns[s_path] = 0 # Initialize if missing, though this indicates an issue
                 path_level.Ns[s_path] += 1

                 # Negate value for the next level up (alternating players)
                 v = -v

        # Note: The return value of this function is not directly used by the modified `search` logic anymore,
        # but might be useful if called directly (e.g., at the end of getActionProb).
        # Returning 0 as a neutral default.
        return 0

    def _clear_batch_data(self):
        """Helper function to reset batch lists."""
        self.leaf_nodes = []
        self.leaf_states = []
        self.leaf_depths = []
        self.pending_searches = []
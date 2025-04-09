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
    Includes fix for 'uninitialized Qsa' by initializing stats earlier.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.nodes = defaultdict(TreeLevel)

        # Batch processing parameters
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
        # Assuming board object tracks move count/depth correctly
        try:
             depth = canonicalBoard.move_count
        except AttributeError:
             log.error("Board object does not have 'move_count' attribute needed for depth.")
             # Handle error: maybe estimate depth or use a default like 0
             depth = 0 # Fallback, adjust as needed

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
            log.warning(f"Root node {s} at depth {depth} not found in MCTS statistics after {self.args.numMCTSSims} simulations. Was it terminal immediately?")
             # Fallback: use uniform probability or prior policy if available
            valid_moves = self.game.getValidMoves(canonicalBoard, 1)
            num_valid = np.sum(valid_moves)
            if num_valid > 0:
                 # Check if Ps was computed (e.g., if 1 sim ran and hit batch limit)
                 if s in current_level.Ps:
                     log.warning("Using prior policy Ps as fallback.")
                     return current_level.Ps[s]
                 else:
                     log.warning("Using uniform valid moves as fallback.")
                     return valid_moves / num_valid
            else: # Should be terminal state if no valid moves
                 log.warning("No valid moves for root node, returning zeros.")
                 return np.zeros(action_size) # Or handle terminal case appropriately


        # Clean up previous depth to save memory (optional, consider if memory is tight)
        # Be careful if using this during training phases where state might be revisited
        # if (depth - 1) in self.nodes:
        #     del self.nodes[depth - 1]

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
                # If no visits (e.g., low sims, immediate batching), return uniform over valid
                log.warning("No visits recorded for root node actions (temp=0), returning uniform over valid moves.")
                valid_moves = self.game.getValidMoves(canonicalBoard, 1)
                num_valid = np.sum(valid_moves)
                if num_valid > 0:
                    return valid_moves / num_valid
                else: # Should be terminal state if no valid moves
                    return np.zeros(action_size)

        # Apply temperature scaling to visit counts
        # Use counts.astype(np.float64) temporarily for power to avoid potential overflow/precision issues with large counts/small temps
        scaled_counts = np.power(np.maximum(counts, 0).astype(np.float64), 1.0 / temp)

        # Normalize to get probabilities
        counts_sum = np.sum(scaled_counts)
        if counts_sum > 0:
            probs = (scaled_counts / counts_sum).astype(np.float32)
        else:
            # Fallback if sum is zero (e.g., after temp scaling of small counts)
            log.warning("Sum of counts is zero after temperature application, returning uniform over valid moves.")
            valid_moves = self.game.getValidMoves(canonicalBoard, 1)
            num_valid = np.sum(valid_moves)
            if num_valid > 0:
                probs = (valid_moves / num_valid).astype(np.float32)
            else: # Should be terminal state if no valid moves
                probs = np.zeros(action_size, dtype=np.float32)

        return probs

    def search(self, canonicalBoard, search_path=None):
        """
        Performs one MCTS search iteration with support for batch evaluation.
        Includes fix: Initializes Ns, Nsa, Qsa earlier.

        Args:
            canonicalBoard: Current board state (must be canonical form).
            search_path: List of (state_str, depth, action) tuples tracing the path.

        Returns:
            v: The negated value backup for the parent node. Returns 0 for terminal states
               visited for the first time, None if pending batch, or the actual backed-up value.
        """
        # Initialize search path if this is the root call
        if search_path is None:
            search_path = []

        s = self.game.stringRepresentation(canonicalBoard)
        # Assuming the board object has a way to track its depth/move count
        try:
            depth = canonicalBoard.move_count
        except AttributeError:
             log.error(f"Board object for state {s} lacks 'move_count'. Using depth {len(search_path)} as estimate.")
             depth = len(search_path) # Estimate depth based on path length

        current_level = self.nodes[depth] # Get the dictionary for the current depth

        # 1. Check if terminal state
        if s not in current_level.Es:
            current_level.Es[s] = self.game.getGameEnded(canonicalBoard, 1) # Player 1 perspective

        if current_level.Es[s] != 0:
            # Terminal state reached. Return the game result (negated for the parent).
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
                 # Execution continues below as if we arrived at an *expanded* node.
                 # Pass explicitly, letting the code continue to selection phase.
                pass
            else:
                # Batch is not full, this search path must wait. Return None.
                return None # Signal pending evaluation

        # --- If we reach here, the node 's' is already expanded ---

        # 3. Select the best action using PUCT/UCB
        if s not in current_level.Vs: # Ensure valid moves are cached
            current_level.Vs[s] = self.game.getValidMoves(canonicalBoard, 1)
        valids = current_level.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # Ensure Ns exists for UCB calculation (should exist if Ps exists)
        if s not in current_level.Ns:
            # This case should ideally not happen if expansion logic is correct in _process_batch
            log.warning(f"Node {s} at depth {depth} has Ps but not Ns during selection. Initializing Ns=0.")
            current_level.Ns[s] = 0

        sqrt_Ns_s_plus_eps = math.sqrt(current_level.Ns[s] + EPS) # Precompute for efficiency

        # Iterate through valid actions to find the best one according to UCB
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in current_level.Qsa:
                    # Standard UCB formula for visited actions
                    q_value = current_level.Qsa[(s, a)]
                    n_sa = current_level.Nsa[(s, a)]
                    u = q_value + self.args.cpuct * current_level.Ps[s][a] * \
                        sqrt_Ns_s_plus_eps / (1 + n_sa)
                else:
                    # Initialize UCB for unvisited actions (Q=0 implicitly)
                    # Rely only on prior probability and parent visit count
                    u = self.args.cpuct * current_level.Ps[s][a] * sqrt_Ns_s_plus_eps
                    # Note: Nsa is implicitly 0, so denominator is (1 + 0) = 1

                if u > cur_best:
                    cur_best = u
                    best_act = a

        # Check if a valid action was found
        if best_act == -1:
             log.error(f"MCTS: No valid action chosen for state {s} at depth {depth}. Valids: {valids}. Ps[s]: {current_level.Ps[s] if s in current_level.Ps else 'Not Expanded'}. Ns[s]: {current_level.Ns.get(s, 'N/A')}")
             valid_indices = np.where(valids == 1)[0]
             if len(valid_indices) > 0:
                 log.warning("Choosing a random valid action as fallback.")
                 best_act = np.random.choice(valid_indices)
             else:
                 log.error("No valid moves available, but not detected as terminal state prior to selection.")
                 # This indicates a contradiction, return 0 but investigate game logic
                 return 0 # Default return value if truly stuck


        a = best_act

        # --- *** FIX IMPLEMENTED HERE (PART 1) *** ---
        # Increment state visit count *now* because we are taking action 'a' from state 's'
        current_level.Ns[s] += 1 # Increment Ns[s] *before* recursion

        # Ensure Qsa/Nsa entries exist for the chosen path *before* recursing or backpropagating later
        # Initialize with Q=0, N=0 if it's the first time taking this action from this state.
        if (s, a) not in current_level.Qsa:
            current_level.Qsa[(s, a)] = 0.0
            current_level.Nsa[(s, a)] = 0
        # --- *** END FIX PART 1 *** ---


        # 4. Recurse down the chosen action
        next_board, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_canonicalBoard = self.game.getCanonicalForm(next_board, next_player)

        path_element = (s, depth, a)
        v = self.search(next_canonicalBoard, search_path + [path_element])

        # 5. Backpropagate the result `v` if the recursive call completed (returned a value)
        if v is not None:
            # --- *** FIX IMPLEMENTED HERE (PART 2) *** ---
            # Update Qsa and Nsa using the returned value v.
            # Need the Nsa count *before* this path incremented it for the formula.
            current_Nsa = current_level.Nsa[(s, a)]
            current_Qsa = current_level.Qsa[(s, a)]

            # Weighted average update for Q-value
            current_level.Qsa[(s, a)] = (current_Nsa * current_Qsa + v) / (current_Nsa + 1)
            current_level.Nsa[(s, a)] += 1 # Increment action count AFTER using current_Nsa

            # REMOVED: current_level.Ns[s] += 1 # Ns[s] was incremented *before* recursion
            # --- *** END FIX PART 2 *** ---

            # Return the negated value for the parent node in the recursion
            return -v
        else:
            # The recursive call returned None (pending batch evaluation).
            # Ns[s] was already incremented. Qsa/Nsa were initialized.
            # Just propagate the None signal upwards. The actual backpropagation
            # for this (s,a) edge will happen later in _process_batch_and_backpropagate
            # when the downstream leaf node's value is known.
            return None


    def _process_batch_and_backpropagate(self):
        """
        Process all leaf nodes currently in the batch using the neural network
        and backpropagate the evaluated values 'v' up their respective search paths.
        Includes fix: Does not increment Ns here, assumes Qsa/Nsa exist.
        """
        if not self.leaf_nodes:
            log.debug("Process batch called with no leaf nodes.")
            return # Nothing to process

        # Make copies of current batch data, as it will be cleared
        current_leaf_boards = list(self.leaf_nodes) # Use list() for explicit copy
        current_leaf_states = list(self.leaf_states)
        current_leaf_depths = list(self.leaf_depths)
        current_pending_searches = list(self.pending_searches)

        # Clear global batch lists immediately to allow new searches to add nodes
        self._clear_batch_data()

        # Prepare batch for neural network
        # Assuming board object has an `encode` method or is directly usable
        try:
            # Check if board object has 'encode' method, otherwise assume it's already encoded
            if hasattr(current_leaf_boards[0], 'encode') and callable(getattr(current_leaf_boards[0], 'encode')):
                 batch_encoded = [board.encode() for board in current_leaf_boards]
            else:
                 log.debug("Board object has no 'encode' method, assuming it's already NN input format.")
                 batch_encoded = current_leaf_boards # Assume boards are already numpy arrays etc.
            batch_tensor = torch.FloatTensor(np.array(batch_encoded).astype(np.float32))
        except Exception as e:
            log.error(f"Error encoding/preparing batch boards: {e}", exc_info=True)
            return # Avoid crashing


        # Move tensor to GPU if configured
        if self.args.cuda and torch.cuda.is_available(): # Check cuda flag in args
            try:
                 batch_tensor = batch_tensor.contiguous().cuda(non_blocking=True)
            except Exception as e:
                 log.error(f"Failed to move batch tensor to CUDA: {e}", exc_info=True)
                 # Decide if you want to proceed on CPU or return
                 return # Return for now


        # Perform batch inference
        self.nnet.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            try:
                log_batch_pi, batch_v = self.nnet.model(batch_tensor)
                # Ensure outputs are moved to CPU and converted to numpy
                batch_pi = torch.exp(log_batch_pi).cpu().numpy()
                batch_v = batch_v.cpu().numpy()
            except Exception as e:
                log.error(f"Error during NN inference: {e}", exc_info=True)
                # Handle error: maybe assign default values? For now, return.
                return


        # --- Update MCTS tree nodes with NN results ---
        for i in range(len(current_leaf_boards)):
            board = current_leaf_boards[i] # Use original board object if needed for game calls
            s = current_leaf_states[i]
            depth = current_leaf_depths[i]
            pi, v_nn = batch_pi[i], batch_v[i][0] # Extract policy and value for this node

            current_level = self.nodes[depth]

            # Ensure state exists in Es (should have been checked in search, but double-check)
            if s not in current_level.Es:
                current_level.Es[s] = self.game.getGameEnded(board, 1)

            # Check if node became terminal *while waiting* for batch processing
            if current_level.Es[s] != 0:
                log.warning(f"Node {s} at depth {depth} became terminal while pending NN eval. Using game result {current_level.Es[s]} instead of NN value {v_nn}.")
                # We don't need to store Ps or Vs for terminal nodes.
                # Ensure Ns is initialized if somehow missed (shouldn't happen)
                if s not in current_level.Ns: current_level.Ns[s] = 0
            else:
                # --- Store NN results for non-terminal leaf node ---
                # Get valid moves if not already cached (might be cached from selection phase if batch filled mid-search)
                if s not in current_level.Vs:
                    current_level.Vs[s] = self.game.getValidMoves(board, 1)
                valids = current_level.Vs[s]
                masked_pi = pi * valids # Mask invalid actions

                sum_ps = np.sum(masked_pi)
                if sum_ps > 0:
                    current_level.Ps[s] = masked_pi / sum_ps # Normalize masked policy
                else:
                    # If NN assigns 0 probability to all valid moves, or no valid moves exist (should be terminal)
                    log.warning(f"NN predicted zero probability for all valid moves for non-terminal state {s} at depth {depth}. Ps: {pi}, Valids: {valids}. Using uniform.")
                    num_valid = np.sum(valids)
                    if num_valid > 0:
                        current_level.Ps[s] = valids / num_valid
                    else:
                        # This indicates a contradiction: non-terminal but no valid moves
                        log.error(f"State {s} at depth {depth} is non-terminal (Es={current_level.Es[s]}) but has no valid moves! Check game logic.")
                        current_level.Ps[s] = np.zeros(self.game.getActionSize()) # Assign zeros

                # Initialize visit count for the newly expanded node
                # Ns should only be initialized here. It will be incremented during backprop from parents.
                if s not in current_level.Ns:
                    current_level.Ns[s] = 0
                # Do NOT increment Ns[s] here. It represents visits *to* the node,
                # which are counted when paths arrive *from* parents during backpropagation.


        # --- Backpropagate values through completed search paths ---
        for i in range(len(current_pending_searches)):
            path = current_pending_searches[i]
            leaf_s = current_leaf_states[i]
            leaf_depth = current_leaf_depths[i]
            leaf_level = self.nodes[leaf_depth]

            # Determine the value to backpropagate: game result or NN prediction
            # The value 'v' represents the outcome for the player whose turn it is AT THE LEAF.
            # We negate it for the parent.
            if leaf_s in leaf_level.Es and leaf_level.Es[leaf_s] != 0:
                # Use the actual game outcome if the leaf became terminal
                v = leaf_level.Es[leaf_s]
            else:
                # Use the NN's value prediction for the leaf state
                v = batch_v[i][0]

            # Iterate backwards through the path (state, depth, action) tuples
            # The path includes the leaf: [(s0, d0, a0), (s1, d1, a1), ..., (s_leaf, d_leaf, None)]
            # We reverse all steps *except* the leaf node's entry itself.
            for s_path, depth_path, a_path in reversed(path[:-1]):
                # Negate value for the parent node (alternating players)
                # The value `v` received here is from the perspective of the child state (s_path's child).
                # We need to update the parent's Qsa from the parent's perspective.
                v = -v
                path_level = self.nodes[depth_path]

                # --- *** FIX IMPLEMENTED HERE (PART 3) *** ---
                # Update statistics for the edge (s_path, a_path)
                # Qsa and Nsa keys should exist due to the fix in search()
                if (s_path, a_path) in path_level.Qsa:
                    current_Nsa = path_level.Nsa[(s_path, a_path)]
                    current_Qsa = path_level.Qsa[(s_path, a_path)]

                    path_level.Qsa[(s_path, a_path)] = (current_Nsa * current_Qsa + v) / (current_Nsa + 1)
                    path_level.Nsa[(s_path, a_path)] += 1 # Increment action count
                else:
                    # This block should ideally NOT be reached with the fix. Log an error if it does.
                    log.error(f"CRITICAL: Qsa/Nsa for ({s_path}, {a_path}) at depth {depth_path} was not initialized before batch backprop! Initializing now, but indicates logic flaw.")
                    path_level.Qsa[(s_path, a_path)] = v
                    path_level.Nsa[(s_path, a_path)] = 1
                    # If Qsa/Nsa were missing, Ns likely wasn't incremented correctly either.
                    # This indicates a problem needs deeper investigation.

                # REMOVED: path_level.Ns[s_path] += 1 # Ns was incremented during the downward pass in search()
                # --- *** END FIX PART 3 *** ---


        # Note: Return value isn't strictly necessary for internal logic anymore
        return 0

    def _clear_batch_data(self):
        """Helper function to reset batch lists."""
        self.leaf_nodes = []
        self.leaf_states = []
        self.leaf_depths = []
        self.pending_searches = []
from collections import namedtuple
import numpy as np
import torch

Transition = namedtuple('Transition', ('timestep', 'state', 'action', 'reward', 'nonterminal'))

blank_trans = Transition(0, torch.zeros(84, 84, dtype=torch.uint8), None, 0, False)


class ReplayMemory():
    def __init__(self, device, num_envs, capacity, history_length, gamma, multi_step, priority_weight, priority_exponent):
        self.num_envs = num_envs
        self.device = device
        self.capacity = capacity
        self.history = history_length
        self.gamma = gamma
        self.n = multi_step
        self.priority_weight = priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = priority_exponent
        self.t = [0] * num_envs  # Internal episode timestep counter for each env
        self.transitions = SegmentTree(
            capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, states, actions, rewards, terminals):
        for i, (state, action, reward, terminal) in enumerate(zip(states, actions, rewards, terminals)):
            state = state[-1].mul(255).to(dtype=torch.uint8, device=torch.device(
                'cpu'))  # Only store last frame and discretise to save memory
            self.transitions.append(Transition(self.t[i], state, action, reward, not terminal),
                                    self.transitions.max)  # Store new transition with maximum priority

            self.t[i] = 0 if terminal else self.t[i] + 1 # update timestep counter


    # Returns a (self.history + n)-step transition
    # with blank states where appropriate
    # self.history = count of adjacent transitions to update on (default 4)
    def _get_transition(self, idx):
        transition = np.array([None] * (self.history + self.n))
        transition[self.history - 1] = self.transitions.get(idx)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            if transition[t + 1].timestep == 0:
                # sets transitions before timestep 0 to blank
                # (histories starting before 0 should not be calculated)
                transition[t] = blank_trans
            else:
                hist_idx = idx + (t + 1 - self.history) * self.num_envs
                transition[t] = self.transitions.get(hist_idx)

                if transition[t] is None:
                    print(f'hist none: {t}, {transition[t + 1].timestep}')
                    print(self.transitions.data[hist_idx:hist_idx + 4])

        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            if transition[t - 1].nonterminal:
                transition[t] = self.transitions.get(idx + (t + 1 - self.history) * self.num_envs)
                if transition[t] is None:
                    print(f'n-step none: {t}, {transition[t - 1].timestep}')
                    print(self.transitions.data[hist_idx % self.transitions.size:(hist_idx + 4) % self.transitions.size])
            else:
                transition[t] = blank_trans  # sets transitions after terminal to blank
        return transition

    def _get_sample_from_segment(self, segment, i):
        valid = False
        while not valid:
            sample = np.random.uniform(i * segment,
                                       (i + 1) * segment)  # Uniformly sample an element from within a segment
            prob, idx, tree_idx = self.transitions.find(
                sample)  # Retrieve sample from tree with un-normalised probability
            # Resample if transition straddled current index or probablity 0
            if (self.transitions.index - idx) % self.capacity > self.n and (
                idx - self.transitions.index) % self.capacity >= self.history and prob != 0:
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        # Retrieve all required transition data (from t - h to t + n)
        transition = self._get_transition(idx)
        # Create un-discretised state and nth next state
        state = torch.stack([trans.state for trans in transition[:self.history]]).to(dtype=torch.float32,
                                                                                     device=self.device).div_(255)
        next_state = torch.stack([trans.state for trans in transition[self.n:self.n + self.history]]).to(
            dtype=torch.float32, device=self.device).div_(255)
        # Discrete action to be used as index
        action = torch.tensor([transition[self.history - 1].action], dtype=torch.int64, device=self.device)
        # Calculate truncated n-step discounted return R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        R = torch.tensor([sum(self.gamma ** n * transition[self.history + n - 1].reward for n in range(self.n))],
                         dtype=torch.float32, device=self.device)
        # Mask for non-terminal nth next states
        nonterminal = torch.tensor([transition[self.history + self.n - 1].nonterminal], dtype=torch.float32,
                                   device=self.device)

        return prob, idx, tree_idx, state, action, R, next_state, nonterminal

    def sample(self, batch_size):
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        segment_size = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        batch = [self._get_sample_from_segment(segment_size, i) for i in range(batch_size)]  # Get batch of valid samples
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals = zip(*batch)
        states, next_states, = torch.stack(states), torch.stack(next_states)
        actions, returns, nonterminals = torch.cat(actions), torch.cat(returns), torch.stack(nonterminals)
        probs = np.array(probs, dtype=np.float32) / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32,
                               device=self.device)  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        [self.transitions.update(idx, priority) for idx, priority in zip(idxs, priorities)]

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        # Create stack of states
        state_stack = [None] * self.history
        state_stack[-1] = self.transitions.data[self.current_idx].state
        prev_timestep = self.transitions.data[self.current_idx].timestep
        for t in reversed(range(self.history - 1)):
            if prev_timestep == 0:
                state_stack[t] = blank_trans.state  # If future frame has timestep 0
            else:
                state_stack[t] = self.transitions.data[self.current_idx + t - self.history + 1].state
                prev_timestep -= 1
        state = torch.stack(state_stack, 0).to(dtype=torch.float32, device=self.device).div_(
            255)  # Agent will turn into batch
        self.current_idx += 1
        return state


# Segment tree data structure where parent node values are sum/max of children node values
# children nodes are the transitions
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.sum_tree = np.zeros((2 * size - 1,),
                                 dtype=np.float32)  # Initialise fixed size tree with all (priority) zeros
        self.data = np.array([None] * size)  # Wrap-around cyclic buffer
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Propagates value up tree given a tree index
    def _propagate(self, index, value):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate(parent, value)

    # Updates value given a tree index
    def update(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate(index, value)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self.update(self.index + self.size - 1, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of a value in sum tree
    def _retrieve(self, index, value):
        left, right = 2 * index + 1, 2 * index + 2
        if left >= len(self.sum_tree):
            return index
        elif value <= self.sum_tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.sum_tree[left])

    # Searches for a value in sum tree and returns value, data index and tree index
    def find(self, value):
        index = self._retrieve(0, value)  # Search for index of item from root
        data_index = index - self.size + 1
        return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]

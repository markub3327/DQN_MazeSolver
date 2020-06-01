import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.max_size = int(max_size)
        self.buffer = deque(maxlen=self.max_size)
    
    def add(self, transition):
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False, dtype=np.float32))
            action.append(np.array(a, copy=False, dtype=np.int))
            reward.append(r)
            next_state.append(np.array(s_, copy=False, dtype=np.float32))
            done.append(d)

        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.buffer_size = 1500000
        # experience tuple consists of number of S + a + Sp + r + done = \
        # 8 + 1 + 8 + 1 + 1 = 19 elements
        self.exp_size = 19
        self.buffer = np.zeros((self.buffer_size, self.exp_size))
        self.current_ind = 0
        self.total_added = 0
    def get_count(self):
        return min(self.total_added, self.buffer_size)
    def add(self, s, a, sp, r, done):
        if self.current_ind >= self.buffer_size:
            self.current_ind = 0
#         import ipdb; ipdb.set_trace()
#         print (np.array([s]).shape, ' | ', self.buffer[self.current_ind].shape)
        self._assign_buffer(s, a, sp, r, done)
        self.current_ind += 1
        self.total_added += 1
    def _assign_buffer(self, s, a, sp, r, done):
        self.buffer[self.current_ind][0:8] = s
        self.buffer[self.current_ind][8] = a
        self.buffer[self.current_ind][9:17] = sp
        self.buffer[self.current_ind][17] = r
        self.buffer[self.current_ind][18] = done
#         print ('---> s, a, sp, r: ', self.buffer[self.current_ind][0:8], self.buffer[self.current_ind][8], 
#               self.buffer[self.current_ind][9:17], self.buffer[self.current_ind][17], self.buffer[self.current_ind][18])
    def sample(self):
        if self.get_count() < self.batch_size:
            batch = np.sample(self.buffer, self.get_count())
        else:
            idx = np.random.choice(self.get_count(), self.batch_size, replace=False)
            batch = self.buffer[idx, :]
#             print ('batch size: ', batch.shape)
        s = batch[:,0:8]
        a = batch[:,8]
        sp = batch[:,9:17]
        r = batch[:,17]
        done = batch[:,18]
        return s, a, sp, r, done
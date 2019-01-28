import numpy as np
import random
import h5py
import os
from datetime import datetime
from DQN import DQN
from keras.models import load_model

class DDQN:
    def __init__(self, env, gamma=0.99, alpha=0.0015):
        self.save_directory = './saved/' + datetime.now().strftime("%d_%m_%y_%HH_%MM_%SS") + '/'
        self.mainQN_save_path = self.save_directory + 'mainQN.h5'
        self.targetQN_save_path = self.save_directory + 'targetQN.h5'
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        self.load_dir = ''
        self.gamma = gamma
        # not using alpha
        self.env = env
        self.no_action = env.action_space.n
        self.no_obs_space = env.observation_space.shape[0]
        self.mainQN = DQN(self.no_action, self.no_obs_space, alpha).build()
        self.targetQN = DQN(self.no_action, self.no_obs_space, alpha).build()
        pass
    def train_main(self, s, a, sp, r, done):
        # train batches of experiences
        targetQs = self.compute_targetQs(s, a, sp, r, done)
        loss = self.mainQN.train_on_batch(s, targetQs)
        return loss
    
    def _reshape(self, s):
        return np.atleast_2d(s)
    
    def compute_targetQs(self, s_batch, a_batch, sp_batch, r_batch, done):
        bs = len(s_batch)
        idx = np.arange(bs, dtype=np.int)
        targetQs = np.zeros((bs, self.no_action))
        targetQs = self.mainQN.predict_on_batch(s_batch)
        discounted_Qs = self.targetQN.predict_on_batch(sp_batch)
#         print ('discounted_Qs shape: ', discounted_Qs.shape)
        targetQs[idx, a_batch.astype(int)] = r_batch + (np.logical_not(done))*(self.gamma*np.max(discounted_Qs, axis=1))
#         print (targetQs)
        return targetQs

    def test_targetQs(self, s, a, sp, r, done):
        batch_size = len(s)
        targetQs = np.zeros((batch_size, self.no_action))
        correct = 0.0
        for i in range(batch_size):
            targetQs[i] = self.mainQN.predict(self._reshape(s[i]))
#             print ('--> targetQs: ', targetQs[i], ' | state: ', s[i], ' | action: ', a[i], ' | r: ', r[i])
            # use target QN to estimate discounted reward for s'
            discounted_Qs = self.targetQN.predict(self._reshape(sp[i]))
            targetQs[i, int(a[i])] = r[i]
            if not done[i]:
                targetQs[i, int(a[i])] += self.gamma * np.max(discounted_Qs)
        return targetQs
        
    def train_target(self):
        # update target DQN weights
        main_weights = self.mainQN.get_weights()
        self.targetQN.set_weights(main_weights)
    def get_save_path(self):
        return self.mainQN_save_path
    def get_load_dir(self):
        return self.load_dir
    def set_load_dir(self, path):
        self.load_dir = path
    def save(self, episode):
        self.mainQN.save(self.mainQN_save_path)
        self.targetQN.save(self.targetQN_save_path)
        print ('--> saved mainQN to: ', self.mainQN_save_path)
    def load(self):
        self.mainQN = load_model(self.load_dir + 'mainQN.h5')
        self.targetQN = load_model(self.load_dir + 'targetQN.h5')
        print ('--> loaded mainQN to: ', self.load_dir)
    def get_action(self, eps, s):
        # choose action to take according to greedy-epilson using main QN
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            # select action from main DQN if not within epsilon
            opt_action = np.argmax(self.mainQN.predict(self._reshape(s)))
            return opt_action
    
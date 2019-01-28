import gym
from gym import wrappers
import numpy as np
from datetime import datetime
import os
import time
from DDQN import DDQN
from ReplayBuffer import ReplayBuffer

class LunarLander:
    def __init__(self, annealing_size=150, alpha=0.0015, batch_size=50, update_step=2, load_weights=False, load_weight_dir=''):
        # define epilson greedy hyperparameters
        self.start_eps = 0.999
        self.end_eps = 0.001
        self.annealing_size = annealing_size
        self.env = gym.make('LunarLander-v2')
        self.env._max_episode_steps = 1000
        self.train_episodes = 2000
        self.batch_size = batch_size
        self.ddqn = DDQN(self.env, gamma=0.99, alpha=alpha)
        self.update_step = update_step
        self.monitor_path = './monitor/' + datetime.now().strftime("%d_%m_%y_%HH_%MM_%SS") + '/'
        if not os.path.exists(self.monitor_path):
            os.makedirs(self.monitor_path)
        self.result_path = './result/' + datetime.now().strftime("%d_%m_%y_%HH_%MM_%SS") + '/'
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.result_train_path = self.result_path + 'ddqn_train_' + datetime.now().strftime("%d_%m_%y_%HH_%MM_%SS") + '.txt'
        self.result_test_path = self.result_path + 'ddqn_test_' + datetime.now().strftime("%d_%m_%y_%HH_%MM_%SS") + '.txt'
        if load_weights:
            if not load_weight_dir:
                raise ValueError('Load weight path must be specified if load_weights is True')
            if not os.path.exists(load_weight_dir):
                raise ValueError('Load weight path must exist')
            # weight_file = Path(load_weight_path)
            # if not weight_file.exists():
            #     raise ValueError('Load weight path doesn"t exit')
            self.ddqn.set_load_dir(load_weight_dir)
            print('--> loading weights at:', self.ddqn.get_load_dir())
            self.ddqn.load()
    def start_record(self, render=False):
        if not render:
            self.env.render(close=True)
        self.env = wrappers.Monitor(self.env, self.monitor_path, force=True)
    def end_record(self, upload_key):
        self.env.close()
        gym.upload(self.monitor_path, api_key=upload_key)
    def train_n_record(self):
        self.start_record()
        self.train()
        self.end_record()
    def test(self):
        print('testing trained model')
        # play 100 episodes 
        file = open(self.result_test_path, 'w+')
        file.write('Episodes\tReward\tIterations\n')
        for n in range(100):
            ep_reward, i = 0, 0
            s = self.env.reset()
            done = False
            while not done:
                # set -1 as eps forces using model predictions
                a = self.ddqn.get_action(-1, s)
                sp, r, done, info = self.env.step(a)
                ep_reward += r
                i += 1
                s = sp
            file.write('{}\t{}\t{}\n'.format(n, ep_reward, i))
        file.close()
        print('completed testing')
            
    def train(self):
        file = open(self.result_train_path, 'w+')
        file.write('Episodes\tReward\tAvgReward\tIterations\tLoss\tTime\n')
        buffer = ReplayBuffer(self.batch_size)
        eps = self.start_eps
        total_rewards = np.zeros(self.train_episodes)
        # initialize target network weights to main weights
        self.ddqn.train_target()
        start_time = time.time()
        for n in range(self.train_episodes):
            ep_reward, i, loss, done = 0, 0, 0, False
            if eps > self.end_eps:
                eps -= (self.start_eps - self.end_eps) / self.annealing_size 
            s = self.env.reset()
            while not done:
                # get action using annealing epsilon greedy
                a = self.ddqn.get_action(eps, s)
                sp, r, done, info = self.env.step(a)
                ep_reward += r
                i += 1
                # add experience to replay buffer
                buffer.add(s, a, sp, r, done)
                # update current state
                s = sp
                if buffer.get_count() > self.batch_size:
                    # sample from buffer if there's enough experience in replay buffer
                    s_n, a_n, sp_n, r_n, done_n = buffer.sample()
                    # train batch main DQN
                    loss += self.ddqn.train_main(s_n, a_n, sp_n, r_n, done_n)
                if i % self.update_step == 0:
                    # train target DQN every so often
                    self.ddqn.train_target()
                # save network periodically
            total_rewards[n] = ep_reward
            # get average reward of past 100 rewards
            avg_reward = total_rewards[max(0, n - 100):n+1].mean()
            file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(n, ep_reward, avg_reward, i, loss, time.time() - start_time))
            if n % 50 == 0:
                # save network every 50 steps
                print ("--> saving network at training ep: ", n)
                self.ddqn.save(n)
            if n % 10 == 0:
                # for debug purposes
                print('--> episode: ', n, ' | iterations: ', i, ' | ep reward: ', ep_reward, ' | avg reward: ', avg_reward, ' | loss: ', loss)
            # break if reward is above threshold
            if avg_reward > 200:
                print ('--> completed training at eps: ', n)
                break
        file.close()
            
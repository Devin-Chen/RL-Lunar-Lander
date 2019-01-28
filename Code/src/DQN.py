from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

class DQN:
    def __init__(self, no_actions, obs_size, alpha=0.0015):
        self.no_actions = no_actions
        self.obs_size = obs_size
        self.alpha = alpha
        return
    def build(self):
        model = Sequential()
        model.add(Dense(60, activation='relu', input_shape=(self.obs_size,)))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(self.no_actions, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.alpha))
        return model
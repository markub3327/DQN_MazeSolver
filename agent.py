import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import NoisyDense, Input, PReLU
from tensorflow.keras.utils import plot_model

class Agent:
    def __init__(self, state_dim=None, action_dim=None, hidden=None, lr=0.001, fileName=None):
        self.model = self.create_network(state_dim, action_dim, hidden, fileName, lr)
        self.target_model = self.create_network(state_dim, action_dim, hidden, fileName, lr)

        self.target_train(1.0)

    def create_network(self, state_dim, action_dim, hidden, fileName, lr):
        if (fileName == None):
            # vstupna vsrtva pre state
            state_input = Input(shape=state_dim)

            l = state_input
            for i in range(len(hidden)):
                l = NoisyDense(hidden[i], activation='swish', kernel_initializer='he_uniform')(l)

            # vystupna vrstva   -- musi byt linear ako posledna vrstva pre regresiu Q funkcie (-nekonecno, nekonecno)!!!
            output = NoisyDense(action_dim, activation='linear', kernel_initializer='glorot_uniform')(l)

            # Vytvor model
            model = Model(inputs=state_input, outputs=output)

            # Skompiluj model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

            model.summary()

            print("Created successful")
        else:
            model = tf.keras.models.load_model(fileName, compile=False)

            # Skompiluj model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

            model.summary()

            print("Loaded successful")

        return model
    
    def predict(self, state):
        return self.model.predict(state)[0]
    
    def reset_noise_target(self):
        for l in self.target_model.layers[1:]:
            l.reset_noise()

    def reset_noise(self):
        for l in self.model.layers[1:]:
            l.reset_noise()

    def remove_noise(self):
        for l in self.model.layers[1:]:
            l.remove_noise()

    def train(self, replay_buffer, batch_size, gamma, tau):
        if len(replay_buffer.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # predikuj akcie pre stavy            
        targets = self.model.predict(states)
        #print(targets, targets.shape)

        # reset Q target net's noise params
        self.reset_noise_target()

        # predikuj buduce akcie podla target siete
        Q_futures = self.target_model.predict(next_states).max(axis=1)
        #print(Q_futures, Q_futures.shape)

        # vypocitaj TD error
        targets[(np.arange(batch_size), actions)] = rewards + ((1-dones) * gamma * Q_futures)
        #print(targets, targets.shape)

        # pretrenuj model
        history = self.model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0)

        # pretrenuj target siet
        self.target_train(tau)

        return history.history['loss']

    def target_train(self, tau):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        self.target_model.set_weights(target_weights)

    def save_plot(self, path='model.png'):
        plot_model(self.model, path)

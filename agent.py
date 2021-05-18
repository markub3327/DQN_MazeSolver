import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model
from noisy_dense import NoisyDense

class Agent:
    def __init__(self, state_dim=None, action_dim=None, hidden=None, lr=0.001, fileName=None):
        self.model = self.create_network(state_dim, action_dim, hidden, fileName, lr)
        self.target_model = self.create_network(state_dim, action_dim, hidden, fileName, lr)

        self._update_target(self.model, self.target_model, tau=tf.constant(1.0))

    def create_network(self, state_dim, action_dim, hidden, fileName, lr):
        if (fileName == None):
            # vstupna vsrtva pre state
            state_input = Input(shape=state_dim)

            l = state_input
            for i in range(len(hidden)):
                l = NoisyDense(hidden[i], activation='relu', kernel_initializer='he_uniform', use_factorised=False)(l)

            # vystupna vrstva   -- musi byt linear ako posledna vrstva pre regresiu Q funkcie (-nekonecno, nekonecno)!!!
            output = NoisyDense(action_dim, activation='linear', kernel_initializer='glorot_uniform', use_factorised=False)(l)

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

            #model.summary()

            print("Loaded successful")

        return model
    
    @tf.function
    def predict(self, state):
        return tf.squeeze(self.model(tf.expand_dims(state, axis=0)), axis=0)            # remove batch_size dim
    
    def reset_noise_target(self):
        for l in self.target_model.layers[1:]:
            l._reset_noise()

    def reset_noise(self):
        for l in self.model.layers[1:]:
            l._reset_noise()

    def remove_noise(self):
        for l in self.model.layers[1:]:
            l._remove_noise()

    def train(self, replay_buffer, batch_size, gamma, tau):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # predikuj akcie pre stavy            
        targets = self.model(states).numpy()
        #print(targets, targets.shape)

        # reset Q target net's noise params
        self.reset_noise_target()

        # predikuj buduce akcie podla target siete
        Q_futures = self.target_model(next_states)
        Q_futures = tf.reduce_max(Q_futures, axis=1)
        #print(Q_futures, Q_futures.shape)

        # vypocitaj TD error
        targets[(np.arange(batch_size), actions)] = rewards + ((1-dones) * gamma * Q_futures)
        #print(targets, targets.shape)

        # pretrenuj model
        loss = self.model.train_on_batch(states, targets)

        # pretrenuj target siet
        self._update_target(self.model, self.target_model, tau=tf.constant(tau))

        return loss

    @tf.function
    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(net.trainable_variables, net_targ.trainable_variables):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    def save_plot(self, path='model.png'):
        plot_model(self.model, path)

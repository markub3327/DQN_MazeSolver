import tensorflow as tf
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.utils import plot_model
from tensorflow_addons.layers.noisy_dense import NoisyDense

class AgentModel(Model):

    def __init__(self, hidden, action_dim):
        super(AgentModel, self).__init__()
        
        self.hidden_layer = []
        for i in range(len(hidden)):
            self.hidden_layer.append(NoisyDense(hidden[i], activation='relu', kernel_initializer='he_uniform'))

        # vystupna vrstva   -- musi byt linear ako posledna vrstva pre regresiu Q funkcie (-nekonecno, nekonecno)!!!
        self.q_values = Dense(action_dim, activation='linear', kernel_initializer='glorot_uniform')

    def call(self, inputs, reset_noise, remove_noise):
        x = self.hidden_layer[0](inputs)
        for i in range(1, len(self.hidden_layer)):
            x = self.hidden_layer[i](x, reset_noise=reset_noise, remove_noise=remove_noise)

        return self.q_values(x)

class Agent:
    def __init__(self, state_dim=None, action_dim=None, hidden=None, lr=0.001, fileName=None):
        self.model = self.create_network(state_dim, action_dim, hidden, fileName, lr)
        self.target_model = self.create_network(state_dim, action_dim, hidden, fileName, lr)

        self._update_target(self.model, self.target_model, tau=tf.constant(1.0))

    def create_network(self, state_dim, action_dim, hidden, fileName, lr):
        if (fileName == None):
            model = AgentModel(hidden, action_dim)

            print("Created successful")
        else:
            model = tf.keras.models.load_model(fileName, compile=False)

            print("Loaded successful")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
    
        return model

    #@tf.function
    def predict(self, state, reset_noise, remove_noise):
        return tf.squeeze(self.model(tf.expand_dims(state, axis=0), reset_noise=reset_noise, remove_noise=remove_noise), axis=0)            # remove batch_size dim

    def train(self, replay_buffer, batch_size, gamma, tau):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # predikuj akcie pre stavy            
        Q_targets = self.model(states, reset_noise=False, remove_noise=False).numpy()

        # predikuj buduce akcie podla target siete
        Q_futures = self.target_model(next_states, reset_noise=True, remove_noise=False)
        Q_futures = tf.reduce_max(Q_futures, axis=1)
        #print(Q_futures, Q_futures.shape)

        # vypocitaj TD error
        Q_targets[(np.arange(batch_size), actions)] = rewards + ((1-dones) * gamma * Q_futures)

        # pretrenuj model
        with tf.GradientTape() as tape:
            q_vals = self.model(states, reset_noise=False, remove_noise=False)
            loss = self.loss_fn(
                y_true=Q_targets, y_pred=q_vals
            )

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # pretrenuj target siet
        self._update_target(self.model, self.target_model, tau=tf.constant(tau))

        return loss

    @tf.function
    def _update_target(self, net, net_targ, tau):
        for source_weight, target_weight in zip(net.trainable_variables, net_targ.trainable_variables):
            target_weight.assign(tau * source_weight + (1.0 - tau) * target_weight)

    def save_plot(self, path='model.png'):
        plot_model(self.model, path)

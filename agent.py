from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, PReLU, GaussianNoise
from wandb.keras import WandbCallback
import numpy as np

class Agent:
    def __init__(self, state_dim, action_dim, fileName=None):
        self.model = self.create_network(state_dim, action_dim, fileName)
        self.target_model = self.create_network(state_dim, action_dim, fileName)

        self.target_train(1.0)

    def create_network(self, state_dim, action_dim, fileName, lr=0.001):
        if (fileName == None):
            # vstupna vsrtva pre state
            state_input = Input(shape=state_dim)

            l1 = Dense(24, use_bias=True, kernel_initializer='he_uniform')(state_input)
            l1 = PReLU(alpha_initializer='zeros')(l1)

            l2 = Dense(48, use_bias=True, kernel_initializer='he_uniform')(l1)
            l2 = PReLU(alpha_initializer='zeros')(l2)

            l3 = Dense(24, use_bias=True, kernel_initializer='he_uniform')(l2)
            l3 = PReLU(alpha_initializer='zeros')(l3)

            # vystupna vrstva   -- musi byt linear ako posledna vrstva pre regresiu Q funkcie (-nekonecno, nekonecno)!!!
            output = Dense(action_dim, activation='linear', use_bias=True, kernel_initializer='glorot_uniform')(l3)

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
    
    def train(self, replay_buffer, batch_size=32, gamma=0.99):
        if len(replay_buffer.buffer) < batch_size: 
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        #print(states.shape)
        #print(actions.shape)
        #print(rewards.shape)
        #print(next_states.shape)
        #print(dones.shape)

        # predikuj akcie pre stavy            
        targets = self.model.predict(states)
        #print(targets, targets.shape)

        # predikuj buduce akcie podla target siete
        Q_futures = self.target_model.predict(next_states).max(axis=1)
        #print(Q_futures, Q_futures.shape)

        # vypocitaj TD error
        targets[(np.arange(batch_size), actions)] = rewards + ((1-dones) * gamma * Q_futures)
        #print(targets, targets.shape)
        
        # pretrenuj model
        self.model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0, callbacks=[WandbCallback(log_weights=True)])

        # pretrenuj target siet
        self.target_train(0.01)

    def target_train(self, tau):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        self.target_model.set_weights(target_weights)


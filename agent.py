from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, NoisyDense
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

            l1 = NoisyDense(128, activation='swish')(state_input)
            l2 = NoisyDense(128, activation='swish')(l1)

            # vystupna vrstva   -- musi byt linear ako posledna vrstva pre regresiu Q funkcie (-nekonecno, nekonecno)!!!
            output = NoisyDense(action_dim, activation='linear')(l2)

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
    
    def train(self, replay_buffer, batch_size=32):
        if len(replay_buffer.buffer) < batch_size: 
            return

        states, actions, rewards, next_states, dones, gammas = replay_buffer.sample(batch_size)
        #print(states.shape)
        #print(actions.shape)
        #print(rewards.shape)
        #print(next_states.shape)
        #print(dones.shape)
        #print(gammas.shape)

        # predikuj akcie pre stavy            
        targets = self.model.predict(states)
        #print(targets, targets.shape)

        # vygeneruj sum pre target siet
        for i in range(len(self.target_model.layers)):
            l = self.target_model.layers[i]
            if 'noisy_dense' in l.name:
                l.reset_noise()  

        # predikuj buduce akcie podla target siete
        Q_futures = self.target_model.predict(next_states).max(axis=1)
        #print(Q_futures, Q_futures.shape)

        # vypocitaj TD error
        targets[(np.arange(batch_size), actions)] = rewards + ((1-dones) * gammas * Q_futures)
        #print(targets, targets.shape)
        
        # pretrenuj model
        self.model.fit(states, targets, batch_size=batch_size, epochs=1, verbose=0, callbacks=[WandbCallback()])

        # pretrenuj target siet
        self.target_train(0.01)

    def target_train(self, tau):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        self.target_model.set_weights(target_weights)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        for i in range(len(self.model.layers)):
            l = self.model.layers[i]
            if 'noisy_dense' in l.name:
                l.reset_noise()  

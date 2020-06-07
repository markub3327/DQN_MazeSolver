from Prostredie.Prostredie import Prostredie
from Prostredie.EnvItem import *
from agent import Agent
from replaybuffer import ReplayBuffer
import time
import numpy as np
import sys
import os
import wandb

def main(test=False):
    try:
        wandb.init(project="dqn_maze")

        if (test == False):
            epsilon = 1.0
            epsilon_decay = 0.9995
            epsilon_min = 0.01
        else:
            epsilon = 0.0
            
        time_max = 10000
    
        if (test == False):
            a1 = Agent(26, 4)
        else:
            a1 = Agent(26, 4, "model.h5")

        replay_buffer = ReplayBuffer()

        env1 = Prostredie(10, 10, 
        [ 
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 0, 0, 1, 1, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
            0, 1, 1, 1, 1, 1, 0, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 0, 1, 0, 0, 4, 0,
        ])
        
        # Trening agenta
        for episode in range(time_max):
            state = env1.reset()
            state = state.reshape((1, state.shape[0]))
            #print(state, state.shape)

            score = 0.0
            for step in range(50):
                if (episode % 100 == 0):
                    env1.render()

                if np.random.rand() < epsilon:
                    action = env1.sample()
                else:            
                    action = np.argmax(a1.predict(state))
            
                next_state, reward, done = env1.step(action)
                next_state = next_state.reshape((1, next_state.shape[0]))
                #print(next_state, next_state.shape)

                score += reward

                if (test == False):
                    replay_buffer.add((np.squeeze(state), action, reward, np.squeeze(next_state), float(done)))
                    a1.train(replay_buffer)

                if (episode % 100 == 0):
                    print(f"stav: {state}")
                    print(f"akcia: {action}")
                    print(f"odmena: {reward}")
                    print(f"done: {done}")
                    print(f"step: {step+1}")
                    print(f"replay_buffer: {len(replay_buffer.buffer)}")
                    print(f"epsilon: {epsilon}")
                    print(f"epoch: {episode+1}/{time_max}")
                    print(f"score: {score}")

                state = next_state
            
                if (done == True):
                    break
        
            wandb.log({"score": score})

            # zniz podiel nahody na akciach
            if (test == False and epsilon > epsilon_min):
                epsilon *= epsilon_decay

    except KeyboardInterrupt:
        print("Game terminated")
        sys.exit()
    finally:
        if (test == False):
            # Save model to file
            a1.model.save("model.h5")

if __name__ == "__main__":
    main()
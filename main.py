import numpy as np
import sys
import os
import wandb

from Prostredie.Prostredie import Prostredie
from Prostredie.EnvItem import *
from agent import Agent
from replaybuffer import ReplayBuffer

def main(test=False):
    try:
        wandb.init(project="dqn_maze")

        time_max = 10000

        if (test == False):
            a1 = Agent(26, 4)
        else:
            a1 = Agent(26, 4, "model.h5")

        # uloz model do obrazku
        a1.save_plot()

        # vytvor dva datasety pre vypocitanie val_loss
        replay_buffer_train = ReplayBuffer()
        replay_buffer_test = ReplayBuffer()

        env1 = Prostredie(10, 10, 
        [ 
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
            1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
            0, 1, 1, 1, 1, 1, 0, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
            0, 1, 1, 1, 0, 1, 0, 0, 4, 0,
        ])
        
        # Trening agenta
        for episode in range(1,time_max):
            state = env1.reset()
            state = np.expand_dims(state, axis=0)
            #print(state, state.shape)

            score, avg_loss, avg_val_loss = 0.0, 0.0, 0.0

            for step in range(1,100):
                if test == True or (episode % 100 == 0):
                    env1.render()

                # reset Q net's noise params
                a1.reset_noise()

                action = np.argmax(a1.predict(state))
            
                next_state, reward, done = env1.step(action)
                #print(next_state, next_state.shape)

                score += reward

                if (test == False):
                    if (np.random.uniform() <= 0.20):
                        replay_buffer_test.add((np.squeeze(state), action, reward, next_state, float(done)))
                    else:
                        replay_buffer_train.add((np.squeeze(state), action, reward, next_state, float(done)))

                    loss, val_loss = a1.train(replay_buffer_train, replay_buffer_test)
                    avg_loss += loss[0]
                    avg_val_loss += val_loss[0]

                if test == True or (episode % 100 == 0):
                    print(f"stav: {state}")
                    print(f"akcia: {action}")
                    print(f"odmena: {reward}")
                    print(f"done: {done}")
                    print(f"step: {step}")
                    print(f"replay_buffer_train: {len(replay_buffer_train.buffer)}")
                    print(f"replay_buffer_test: {len(replay_buffer_test.buffer)}")
                    print(f"epoch: {episode}/{time_max}")
                    print(f"score: {score}")

                state = np.expand_dims(next_state, axis=0)
 
                if done == True:
                    avg_loss /= step
                    avg_val_loss /= step
                    break
        
            wandb.log({"score": score, "loss": avg_loss, "val_loss": avg_val_loss, "epoch": episode})

    except KeyboardInterrupt:
        print("Game terminated")
        sys.exit()
    finally:
        if (test == False):
            # Save model to file
            a1.model.save("model.h5")

if __name__ == "__main__":
    main()
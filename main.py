import numpy as np
import sys
import os
import wandb
import time
import statistics

from Prostredie.Prostredie import Prostredie
from Prostredie.EnvItem import *
from agent import Agent
from replaybuffer import ReplayBuffer

def main(test=False):
    try:
        # init wandb cloud
        if (test == False):
            wandb.init(project="dqn_maze")

            wandb.config.batch_size = 32
            wandb.config.gamma = 0.95
            wandb.config.h1 = 4
            wandb.config.h2 = 4
            wandb.config.lr = 0.001
            wandb.config.tau = 0.01

        time_max = 1000

        if (test == False):
            a1 = Agent(26, 4, [wandb.config.h1, wandb.config.h2], wandb.config.lr)
        else:
            a1 = Agent(26, 4, [wandb.config.h1, wandb.config.h2], 0.0, "model.h5")

        # uloz model do obrazku
        a1.save_plot()

        # experiences replay buffer
        replay_buffer = ReplayBuffer()

        # generate env
        env1 = Prostredie(10, 10, 
        [ 
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 1, 1, 1, 1, 1, 0, 0,
            1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 1, 0, 1, 0, 1, 0, 0, 1, 0,
            0, 1, 1, 1, 1, 1, 0, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
            0, 1, 1, 1, 0, 1, 0, 0, 4, 0,
        ])
        
        # Trening agenta
        for episode in range(1,time_max+1):
            start_time = time.time()

            state = env1.reset()
            state = np.expand_dims(state, axis=0)

            # reset score
            score, avg_loss = 0.0, 0.0

            for step in range(1,101):
                if test == True:
                    env1.render()

                # reset Q net's noise params
                a1.reset_noise()

                action = np.argmax(a1.predict(state))
            
                next_state, reward, done, info = env1.step(action)

                score += reward

                if (test == False):
                    replay_buffer.add((np.squeeze(state), action, reward, next_state, float(done)))

                    loss = a1.train(replay_buffer, wandb.config.batch_size, wandb.config.gamma, wandb.config.tau)
                    if loss is not None:
                        avg_loss += loss[0]
                else:
                    print(f"stav: {state}")
                    print(f"akcia: {action}")
                    print(f"odmena: {reward}")
                    print(f"done: {done}")
                    print(f"step: {step}")
                    print(f"replay_buffer_train: {len(replay_buffer.buffer)}")
                    print(f"epoch: {episode}/{time_max}")
                    print(f"score: {score}")
                    print(f"apples: {info['apples']}/{env1.count_apple}")
                    print(f"mines: {info['mines']}/{env1.count_mine}")
                
                # critical
                state = np.expand_dims(next_state, axis=0)
 
                if done == True:
                    break

            # statistics
            avg_loss /= step
             
            if (test == False):
                log_dict = {'epoch': episode,
                            'score': score, 
                            'steps': step,
                            'loss': avg_loss,
                            'replay_buffer': len(replay_buffer.buffer),
                            'time': time.time()-start_time,
                            'apple': (info['apples'] / env1.count_apple) * 100.0,
                            'mine': (info['mines'] / env1.count_mine) * 100.0,
                            'end': info['end'] * 100.0}

                wandb.log(log_dict)

    except KeyboardInterrupt:
        print("Game terminated")
        sys.exit()
    finally:
        # Save model to file
        if (test == False):
            a1.model.save("model.h5")

if __name__ == "__main__":
    main()
import numpy as np
import sys
import os
import wandb
import time
import statistics
import argparse

from Prostredie.Prostredie import Prostredie
from Prostredie.EnvItem import *
from agent import Agent
from replaybuffer import ReplayBuffer

def main(test=False):
    try:
        if (test == False):
            # init wandb cloud
            wandb.init(project="dqn_maze")
            
            # hyperparametre
            wandb.config.batch_size = 64
            wandb.config.gamma = 0.98
            wandb.config.h1 = 128
            wandb.config.h2 = 128
            wandb.config.lr = 0.001
            wandb.config.tau = 0.01
            max_episodes = 5000
            max_steps = 100
        else:
            np.random.seed(99)

            max_episodes = 20
            max_steps = 100
            
            # init file
            log_file = open("log/statistics.txt", "w")
            log_file.write("episode;score;step;time;apples;mines;end\n")

        if (test == False):
            a1 = Agent(26, 4, [wandb.config.h1, wandb.config.h2], wandb.config.lr)
        else:
            a1 = Agent(fileName="model.h5")
            a1.remove_noise()

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
            0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
            0, 1, 1, 1, 0, 1, 0, 0, 4, 0
        ])
        
        # Hlavny cyklus hry
        for episode in range(1,max_episodes+1):
            start_time = time.time()

            state = env1.reset(testing=test)

            # reset score
            score, avg_loss = 0.0, 0.0

            for step in range(1,max_steps+1):
                if test == True:
                    env1.render()
                    time.sleep(0.2)
                else:
                    # reset Q net's noise params
                    a1.reset_noise()

                # clovek
                #in_key = input()
                #if in_key == 'w':
                #    action = 1
                #elif in_key == 's':
                #    action = 0
                #elif in_key == 'a':
                #    action = 2
                #elif in_key == 'd':
                #    action = 3
                
                # nahodny agent
                #action = np.random.randint(0, 4)
                
                # neuronova siet
                action = np.argmax(a1.predict(state))
            
                next_state, reward, done, info = env1.step(action)

                score += reward

                if (test == False):
                    replay_buffer.add((state, action, reward, next_state, float(done)))

                    if len(replay_buffer.buffer) >= wandb.config.batch_size:
                        loss = a1.train(replay_buffer, wandb.config.batch_size, wandb.config.gamma, wandb.config.tau)
                        avg_loss += loss
                #else:
                #    print(f"stav: {state}")
                #    print(f"akcia: {action}")
                #    print(f"odmena: {reward}")
                #    print(f"done: {done}")
                #    print(f"step: {step}")
                #    print(f"replay_buffer_train: {len(replay_buffer.buffer)}")
                #    print(f"epoch: {episode}/{max_episodes}")
                #    print(f"score: {score}")
                #    print(f"apples: {info['apples']}/{env1.count_apple}")
                #    print(f"mines: {info['mines']}/{env1.count_mine}")
                
                # critical
                state = next_state
 
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
            else:
                log_file.write(f"{episode};{score};{step};{time.time()-start_time};{(info['apples'] / env1.count_apple) * 100.0};{(info['mines'] / env1.count_mine) * 100.0};{info['end'] * 100.0}\n")

    except KeyboardInterrupt:
        print("Game terminated")
        sys.exit()
    finally:
        # Save model to file
        if (test == False):
            a1.model.save("model.h5")
        else:
            log_file.close()

        env1.f_startPosition.close()
        env1.f_apples.close()
        env1.f_mines.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MazeSolver  --  Double Deep Q network')
    parser.add_argument('--train', dest='training', action='store_true', help='training the model on random mazes')
    parser.add_argument('--test', dest='testing', action='store_true', help='testing the model on mazes')

    args = parser.parse_args()

    print(f'{args.training}, {args.testing}')

    if args.training == True:
        main(False)
    elif args.testing == True:
        main(True)

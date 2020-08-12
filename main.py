import time
import numpy as np
import sys
import os
import wandb

from collections import deque
from Prostredie.Prostredie import Prostredie
from Prostredie.EnvItem import *
from agent import Agent
from replaybuffer import ReplayBuffer

def N_Step_Reward(exp_buffer, discount_rate=0.99):
    state_0, action_0, reward_0 = exp_buffer.popleft()  # vytiahni a vymaz prvy zaznam zlava fronty
    discounted_reward = reward_0                        # znizena odmena z buducich krokov
    gamma = discount_rate                               # prva zlava zacina na DISCOUNT_RATE
    
    for (_, _, r) in exp_buffer:                        # pre zvysnych N skusenosti
        discounted_reward += r * gamma                  # zlavnena buduca odmena ako suma odmien zo skusenosti
        gamma *= discount_rate                          # umocni parameter zlavy podla N krokov    

    #print(discounted_reward, gamma, len(exp_buffer))
    return state_0, action_0, discounted_reward, gamma

def main(test=False):
    try:
        wandb.init(project="dqn_maze")

        time_max = 10000
        start_episode = 100
        n_steps = 5

        if (test == False):
            a1 = Agent(26, 4)
        else:
            a1 = Agent(26, 4, "model.h5")

        replay_buffer = ReplayBuffer()

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
        
        # Initialise deque buffer to store experiences for N-step returns
        exp_buffer = deque()

        # Trening agenta
        for episode in range(time_max):
            state = env1.reset()
            state = np.expand_dims(state, axis=0)
            #print(state, state.shape)

            score = 0.0
            for step in range(50):
                if (episode % 10 == 0):
                    env1.render()

                # pridaj k vaham neuronu novy sum pre lepsie prehladavanie herneho prostredia
                a1.reset_noise()

                if episode < start_episode:
                    action = env1.sample()
                else:
                    # max Q sa povazuje za najvyhodnejsiu akciu
                    action = np.argmax(a1.predict(state))
            
                next_state, reward, done = env1.step(action)
                next_state = np.expand_dims(next_state, axis=0)
                #print(next_state, next_state.shape)

                # scitaj reward za hernu epizodu
                score += reward

                # pridaj s,a,r do bufferu skusenosti agenta 
                exp_buffer.append((state, action, reward))

                # super critic
                state = next_state
            
                # ak je nazbierany dostatocny pocet skusenosti vypocitaj N-krokovu odmenu
                if len(exp_buffer) >= n_steps:
                    state_0, action_0, discounted_reward, gamma = N_Step_Reward(exp_buffer)
                    # pridaj N-krokovu skusenost do replay bufferu
                    replay_buffer.add((np.squeeze(state_0), action_0, discounted_reward, np.squeeze(next_state), float(done), gamma))

                if (test == False):
                    a1.train(replay_buffer)

                if (episode % 10 == 0):
                    print(f"stav: {state}")
                    print(f"akcia: {action}")
                    print(f"odmena: {reward}")
                    print(f"done: {done}")
                    print(f"step: {step+1}")
                    print(f"replay_buffer: {len(replay_buffer.buffer)}")
                    print(f"epoch: {episode+1}/{time_max}")
                    print(f"score: {score}")

                if (done == True):
                    # pouzi vsetky zvysne skusenosti k doplneniu replay bufferu                     
                    while len(exp_buffer) != 0:
                        state_0, action_0, discounted_reward, gamma = N_Step_Reward(exp_buffer)
                        # pridaj N-krokovu skusenost do replay bufferu
                        replay_buffer.add((np.squeeze(state_0), action_0, discounted_reward, np.squeeze(next_state), float(done), gamma))
                        #print(f"discounted_reward = {discounted_reward}, gamma = {gamma}, ExpBuff: {len(exp_buffer)}")
                    break
        
            wandb.log({"score": score})

    except KeyboardInterrupt:
        print("Game terminated")
        sys.exit()
    finally:
        if (test == False):
            # Save model to file
            a1.model.save("model.h5")

if __name__ == "__main__":
    main()
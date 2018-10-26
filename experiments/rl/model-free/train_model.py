import gym
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from baseline import Policy
import torch.optim as optim
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


def select_action(policy, state):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()
    log_prob = torch.Tensor([c.log_prob(action)])
    
    # Add log probability of our chosen action to our history    
    if policy.policy_history.size()[0] != 0:
        policy.policy_history = torch.cat([policy.policy_history, log_prob])
    else:
        policy.policy_history = log_prob
        
    return action

def update_policy(policy, optimizer ):
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards , requires_grad = True)).mul(-1), -1))
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Save and intialize episode history counters
    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = Variable(torch.Tensor(), requires_grad = True)
    policy.reward_episode= []

def train(env, policy, episodes, optimizer):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset() # Reset environment and record the starting state
        done = False       
    
        for time in range(1000):
            action = select_action(policy, state)
            # Step through environment using chosen action
            state, reward, done, _ = env.step(action.data.cpu().numpy())

            # Save reward
            policy.reward_episode.append(reward)
            if done:
                break
        
        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        update_policy(policy, optimizer)

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, time))
            break



def main():

    env = gym.make('CartPole-v1')
    env.seed(1); torch.manual_seed(1);

    #Hyperparameters
    learning_rate = 0.01
    gamma = 0.99
    episodes = 1000

    #Define model to learn policy for the environment
    policy = Policy(env, gamma)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    #Train 
    train(env,policy,episodes,optimizer)

    # Plot
    window = int(episodes/20)
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9,9]);
    rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
    std = pd.Series(policy.reward_history).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(policy.reward_history)),rolling_mean-std, rolling_mean+std, color='orange', alpha=0.2)
    ax1.set_title('CartPole-v1 Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Episode Length')

    ax2.plot(policy.reward_history)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)
    #plt.show()
    plt.savefig('results.png')


if __name__ == '__main__':
    main()



from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch
import argparse

from dqn_agent import Agent


def dqn(env, agent, nfile, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.001, eps_decay=0.985):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    brain_name = env.brain_names[0]
    scores = []  # list containing scores from each episode
    scores_mean = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset()[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            #print(state)
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]
            #next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        scores_mean.append(np.mean(scores_window))

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), nfile)
            break
    return scores, scores_mean

def test_agent(agent, brain_name):
    total_reward =0
    env_info = env.reset()[brain_name]
    state = env_info.vector_observations[0]
    for j in range(200):
        action = agent.act(state)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        total_reward += reward
        done = env_info.local_done[0]
        if done:
            break
    return total_reward

def plot_save_score(scores, file_name):
    v_scores = np.array([range(1, len(scores)+1), scores])
    np.savetxt(file_name, np.transpose(v_scores), delimiter=',')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    fig.savefig("training.pdf", bbox_inches='tight')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Flow
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(train=True)
    parser.set_defaults(test=False)

    # File
    parser.add_argument("-nfile", "--nerualfile", default="checkpoint.pth",
                        help="the file to store the weight of neural network")

    # Options
    parser.add_argument('--ddqn', dest='ddqn', action='store_true',
                         help='double dqn will be executed if take this option, in default original dqn will be adopted')
    parser.set_defaults(test=False)

    args = parser.parse_args()

    print(args.train)

    env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]


    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state


    agent = Agent(state_size=state_size, action_size=action_size, seed=0, double = args.ddqn)

    if args.train :
        scores, scores_mean = dqn(env, agent, args.nerualfile)
        ofile = "score_history.csv"
        plot_save_score(scores_mean, ofile)

    if args.test :
        agent.qnetwork_local.load_state_dict(torch.load(args.nerualfile))
        print("total reward", test_agent(agent, brain_name))


    env.close()
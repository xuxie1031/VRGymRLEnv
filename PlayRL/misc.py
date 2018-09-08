import os
import torch
import numpy as np

def save_checkpoint_ddpg(agent, filename, steps, rewards):
    state = {}

    state['network_dict'] = agent.network.state_dict()
    state['target_network_dict'] = agent.target_network.state_dict()
    state['actor_opt_dict'] = agent.network.actor_opt.state_dict()
    state['critic_opt_dict'] = agent.network.critic_opt.state_dict()
    state['replay'] = agent.replay.get_params()
    state['steps'] = steps
    state['rewards'] = rewards

    torch.save(state, filename)


def load_checkpoint_ddpg(agent, filename):
    assert os.path.isfile(filename)
    state = torch.load(filename)

    agent.network.load_state_dict(state['network_dict'])
    agent.target_network.load_state_dict(state['target_network_dict'])
    agent.network.actor_opt.load_state_dict(state['actor_opt_dict'])
    agent.network.critic_opt.load_state_dict(state['critic_opt_dict'])
    agent.replay.set_params(*state['replay'])
    steps = state['steps']
    rewards = state['rewards']

    return agent, steps, rewards


def save_checkpoint_a2c(agent, filename, steps, rewards):
    state = {}

    state['network_dict'] = agent.network.state_dict()
    state['network_opt'] = agent.optimizer.state_dict()
    state['steps'] = steps
    state['rewards'] = rewards

    torch.save(state, filename)


def load_checkpoint_a2c(agent, filename):
    assert os.path.isfile(filename)
    state = torch.load(filename)

    agent.network.load_state_dict(state['network_dict'])
    agent.optimizer.load_state_dict(state['network_opt'])
    steps = state['steps']
    rewards = state['rewards']

    return agent, steps, rewards


def run_episodes(agent):
    steps = []
    rewards = []

    if agent.config.load_model:
        agent, steps, rewards = load_checkpoint_ddpg(agent, '{0}.pth.tar'.format(agent.task.name))

    epi = 0
    while True:
        reward, step = agent.episode()
        steps.append(step)
        rewards.append(reward)
        save_checkpoint_ddpg(agent, '{0}.pth.tar'.format(agent.task.name), steps, rewards)
        if epi % 20 == 0:
            print('total steps {0}, avg reward {1}'.format(sum(steps), np.mean(rewards[-100:])))


def run_iterations(agent):
    steps = []
    rewards = []

    if agent.config.load_model:
        agent, steps, rewards = load_checkpoint_a2c(agent, '{0}.pth.tar'.format(agent.task.name))

    epi = 0
    while True:
        agent.iteration()
        steps.append(agent.total_steps)
        rewards.append(np.mean(agent.episode_rewards))
        save_checkpoint_a2c(agent, '{0}.pth.tar'.format(agent.task.name), steps, rewards)
        if epi % 20 == 0:
            print('total steps {0}, avg reward {1}'.format(sum(steps), np.mean(rewards[-100:])))
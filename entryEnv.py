#! /usr/bin/env python

import rospy
import threading
from PlayRL import *


class RunRLPlayGround(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)


    def ddpg_low_dim_state(self):
        config = Config()
        config.task_fn = lambda: VRMazeTaskStateContinuous('VRMazeTaskContinuous-State')
        config.network_fn = lambda state_dim, action_dim: DeterministicActorCriticNet(
            state_dim, action_dim,
            actor_body=FCBody(state_dim, (300, 200), gate=torch.tanh),
            critic_body=TwoLayerFCBodyWithAction(state_dim, action_dim, (400, 300), gate=torch.tanh),
            actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
            critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3), gpu=0)

        config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=64)
        config.random_process_fn = lambda action_dim: OrnsteinUhlenbeckProcess(
            size=(action_dim, ), std=LinearSchedule(0.2))

        config.discount = .99
        config.min_memory_size = 64
        config.target_network_mix = 1e-3
        config.load_model = True

        ddpg_low_dim_agent = DDPGAgent(config)
        run_episodes(ddpg_low_dim_agent)

    
    def ddpg_pixel(self):
        config = Config()
        config.task_fn = lambda: VRMazeTaskPixelContinuous('VRMazeTaskContinuous-Image')

        phi_body = DDPGConvBody(in_channels=1)
        config.network_fn = lambda state_dim, action_dim: DeterministicActorCriticNet(
            state_dim, action_dim,
            phi_body = phi_body,
            actor_body=FCBody(phi_body.feature_dim, (50, ), gate=torch.tanh),
            critic_body=OneLayerFCBodyWithAction(phi_body.feature_dim, action_dim, 50, gate=torch.tanh),
            actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
            critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3), gpu=0)

        config.replay_fn = lambda: Replay(memory_size=1000000, batch_size=16)
        config.random_process_fn = lambda action_dim: OrnsteinUhlenbeckProcess(
            size=(action_dim, ), std=LinearSchedule(0.2))
        config.min_memory_size = 64
        config.target_network_mix = 1e-3
        config.load_model = True

        ddpg_pixel_agent = DDPGAgent(config)
        run_episodes(ddpg_pixel_agent)


    def a2c_pixel(self):
        config = Config()
        config.num_workers = 1
        task_fn = lambda name: VRMazeTaskPixelDiscrete(name)
        config.task_fn = lambda: ParallelizedTask(task_fn, 'VRMazeTaskDiscreteA2C-Image', config.num_workers)
        config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=5e-6)
        config.network_fn = lambda state_dim, action_dim: CategoricalActorCriticNet(
            state_dim, action_dim, NatureConvBody(in_channels=1), gpu=0)
        
        config.discount = 0.95
        config.use_gae = False
        config.gae_tau = 0.97
        config.entropy_weight = 0.01
        config.rollout_length = 200
        config.gradient_clip = 0.1
        # config.load_model = True

        a2c_pixel_agent = A2CAgent(config)
        run_iterations(a2c_pixel_agent)


    def dqn_pixel(self):
        config = Config()
        config.task_fn = lambda: VRMazeTaskPixelDiscrete('VRMazeTaskDiscreteDQN-Image')
        config.network_fn = lambda state_dim, action_dim: VanillaNet(action_dim, NatureConvBody(in_channels=1), gpu=0)
        config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=.00025, alpha=.95, eps=.01)
        config.policy_fn = lambda: GreedyPolicy(epsilon=1.0, final_step=1000000, min_epsilon=0.1)
        config.replay_fn = lambda: Replay(memory_size=100000, batch_size=32)
        
        config.discount = 0.99
        config.target_network_update_freq = 1000
        config.exploration_steps = 500000
        config.double_q = False
        config.gradient_clip = 0.1
        # config.load_model = True

        dqn_pixel_agent = DQNAgent(config)
        run_episodes(dqn_pixel_agent)


    def run(self):
        # self.ddpg_low_dim_state()
        # self.ddpg_pixel()
        # self.a2c_pixel()
        self.dqn_pixel()


if __name__ == '__main__':

    rospy.init_node('vrgymrl', anonymous=True)

    rl_thread = RunRLPlayGround()
    rl_thread.daemon = True
    rl_thread.start()
    rospy.spin()
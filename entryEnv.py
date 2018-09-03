#! /usr/bin/env python

import rospy
import threading
from PlayRL import *


class RunRLPlayGround(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)


    def ddpg_low_dim_state(self):
        config = Config()
        config.task_fn = lambda: VRMazeTaskState('VRMazeTask-State')
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

    
    def run(self):
        self.ddpg_low_dim_state()

if __name__ == '__main__':

    rospy.init_node('vrgymrl', anonymous=True)

    rl_thread = RunRLPlayGround()
    rl_thread.daemon = True
    rl_thread.start()
    rospy.spin()
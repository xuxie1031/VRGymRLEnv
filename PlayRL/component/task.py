import rospy
from std_msgs.msg import String

import numpy as np
import time
import json


class VRMazeTaskBase:
    def __init__(self):
        self.pub = rospy.Publisher('/rl_trainer_pub', String, queue_size=1)
        self.sub = rospy.Subscriber('/rl_trainer_sub', String, callback=self.step_callback, queue_size=1)
        self.json_msg = ''
        self.callback_flag = False

        while self.pub.get_num_connections() == 0:
            time.sleep(.001)


    def step_callback(self, msg):
        self.json_msg = msg.data
        self.callback_flag = True


    def wait_execution(self):
        start_t = time.time()
        while not self.callback_flag:   # timeout variable needed
            time.sleep(.000000001)
            end_t = time.time()
            if end_t-start_t > 10.0:     # 20.0s waiting to resend
                print('message loss')
                msg_dict = {'repeat': True}
                msg = json.dumps(msg_dict)

                # self.sub.unregister()
                # time.sleep(1.0)
                # self.sub = rospy.Subscriber('/irl_trainer_sub', String, callback=self.step_callback, queue_size=1)

                start_t = time.time()
                self.pub.publish(msg)

        self.callback_flag = False


    def reset(self):
        pass


    def step(self, state, action):
        pass


class VRMazeTaskState(VRMazeTaskBase):
    def __init__(self, name, state_dim=3, action_dim=3):
        VRMazeTaskBase.__init__(self)
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim     


    def reset(self):
        msg_dict = {'continuousaction': {'reset': True}}
        msg = json.dumps(msg_dict)
        self.pub.publish(msg)

        # wait for execution
        self.wait_execution()

        obj = json.loads(self.json_msg)
        assert 'state' in obj

        return np.asarray([obj['state']['X'], obj['state']['Y'], obj['state']['Z']])


    def step(self, state, action):
        action = action.clip(-1.0, 1.0)
        msg_dict = {'continuousaction': {'action': {'X': float(action[0]), 'Y': float(action[1]), 'Z': float(action[2])}}}
        msg = json.dumps(msg_dict)
        self.pub.publish(msg)

        # wait for execution
        self.wait_execution()

        # print(self.json_msg)
        obj = json.loads(self.json_msg)
        assert 'action' in obj
        assert 'next_state' in obj
        assert 'reward' in obj
        assert 'terminal' in obj

        return np.asarray([obj['next_state']['X'], obj['next_state']['Y'], obj['next_state']['Z']]), float(obj['reward']), int(obj['terminal'])        


class VRMazeTaskPixel(VRMazeTaskBase):
    def __init__(self, name, state_dim=84, action_dim=3):
        VRMazeTaskBase.__init__(self)
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim     


    def reset(self):
        msg_dict = {'continuousimgaction': {'reset': True}}
        msg = json.dumps(msg_dict)
        self.pub.publish(msg)

        # wait for execution
        self.wait_execution()

        obj = json.loads(self.json_msg)
        assert 'state' in obj

        state = np.asarray(obj['state'], dtype=np.float32).reshape((self.state_dim, -1))/255.0

        return np.asarray([state])  # one channel


    def step(self, state, action):
        action = action.clip(-1.0, 1.0)
        msg_dict = {'continuousimgaction': {'action': {'X': float(action[0]), 'Y': float(action[1]), 'Z': float(action[2])}}}
        msg = json.dumps(msg_dict)
        self.pub.publish(msg)

        # wait for execution
        self.wait_execution()

        obj = json.loads(self.json_msg)
        assert 'action' in obj
        assert 'next_state' in obj
        assert 'reward' in obj
        assert 'terminal' in obj

        next_state = np.asarray([obj['next_state']], dtype=np.float32).reshape((self.state_dim, -1))/255.0

        return np.asarray([next_state]), float(obj['reward']), int(obj['terminal'])
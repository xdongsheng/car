

from carla.agent.agent import Agent
from carla.client import VehicleControl
from xds_DQN import DeepQNetwork
from carla.image_converter import depth_to_local_point_cloud, to_rgb_array
import carla.transform as transform




class ForwardAgent(Agent):

    # RL = DeepQNetwork(n_actions=7,
    #                   n_features=image_RGB_real.shape[0],
    #                   learning_rate=0.01, e_greedy=0.9,
    #                   replace_target_iter=100, memory_size=2000,
    #                   e_greedy_increment=0.001, )
    #
    # total_steps = 0
    """
    Simple derivation of Agent Class,
    A trivial agent agent that goes straight
    """

    def run_step(self, measurements, sensor_data, directions, target,rl):
        actions = [-1.0, -0.5, 0, 0.5, 1.0, 0.5, 1.0]
        image_RGB = to_rgb_array(sensor_data['CameraRGB'])
        image_RGB_real = image_RGB.flatten()
        observation=image_RGB_real
        # print(image_RGB_real.shape[0])
        action=rl.choose_action(observation)
        brake1 = 0.0
        steer1 = 0.0
        if (action > 4):
            brake1 = actions[action]
        else:
            steer1 = actions[action]
        control = VehicleControl()
        control.throttle = 0.6
        control.steer = steer1
        control.brake = brake1

        return control,observation,action

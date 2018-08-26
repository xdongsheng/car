# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import abc
import logging
import math
import time

from carla.client import VehicleControl
from carla.client import make_carla_client
from carla.driving_benchmark.metrics import Metrics
from carla.planner.planner import Planner
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError

from . import results_printer
from .recording import Recording
from carla.image_converter import depth_to_local_point_cloud, to_rgb_array

from xds_DQN import DeepQNetwork

def sldist(c1, c2):
    return math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)


class DrivingBenchmark(object):
    """
    The Benchmark class, controls the execution of the benchmark interfacing
    an Agent class with a set Suite.


    The benchmark class must be inherited with a class that defines the
    all the experiments to be run by the agent
    """

    def __init__(
            self,
            city_name='Town01',
            name_to_save='Test',
            continue_experiment=False,
            save_images=False,
            distance_for_success=2.0
    ):

        self.__metaclass__ = abc.ABCMeta

        self._city_name = city_name
        self._base_name = name_to_save
        # The minimum distance for arriving into the goal point in
        # order to consider ir a success
        # 确认是否到达目标
        self._distance_for_success = distance_for_success
        # The object used to record the benchmark and to able to continue after
        # 该对象用于记录基准并能够继续
        self._recording = Recording(name_to_save=name_to_save,
                                    continue_experiment=continue_experiment,
                                    save_images=save_images
                                    )

        # We have a default planner instantiated that produces high level commands
        # 我们有一个实例化的默认计划程序，可以生成高级命令
        self._planner = Planner(city_name)

    def benchmark_agent(self, experiment_suite, agent, client):
        """
        Function to benchmark the agent.
        It first check the log file for this benchmark.
        if it exist it continues from the experiment where it stopped.
        # 用于对代理进行基准测试的功能。首先检查日志文件以获取此基准。如果它存在，则从实验停止。

        Args:
            experiment_suite
            agent: an agent object with the run step class implemented.
            client:


        Return:
            A dictionary with all the metrics computed from the
            agent running the set of experiments.
            # 包含从运行实验集的代理计算的所有度量的字典。
        """

        # Instantiate a metric object that will be used to compute the metrics for
        # the benchmark afterwards.
        # 实例化一个度量对象，该对象随后将用于计算基准的度量标准。
        metrics_object = Metrics(experiment_suite.metrics_parameters,
                                 experiment_suite.dynamic_tasks)

        # Function return the current pose and task for this benchmark.
        # 函数返回此基准的当前位置和任务。
        start_pose, start_experiment = self._recording.get_pose_and_experiment(
            experiment_suite.get_number_of_poses_task())

        print(start_pose,start_experiment)

        logging.info('START')
        print('START')
        # 测试RL的位置
        RL = DeepQNetwork(n_actions=7,
                          n_features=100800,
                          learning_rate=0.01, e_greedy=0.9,
                          replace_target_iter=100, memory_size=2000,
                          e_greedy_increment=0.001, )

        total_steps = 0
        rl_episode = 0


        # for experiment in experiment_suite.get_experiments()[int(start_experiment):]:
        while True:
            rl_episode += 1
            a = experiment_suite.get_experiments()[int(start_experiment):]
            #for a in experiment_suite.get_experiments()[int(start_experiment):]:
            experiment = a[0]

            print('experiment')
            print(experiment)
            positions = client.load_settings(
                experiment.conditions).player_start_spots

            self._recording.log_start(experiment.task)

            for pose in experiment.poses[start_pose:]:
                for rep in range(experiment.repetitions):

                    start_index = pose[0]
                    end_index = pose[1]

                    client.start_episode(start_index)
                    # Print information on
                    logging.info('======== !!!! ==========')
                    logging.info(' Start Position %d End Position %d ',
                                 start_index, end_index)
                    print(start_index,end_index)

                    self._recording.log_poses(start_index, end_index,
                                              experiment.Conditions.WeatherId)

                    # Calculate the initial distance for this episode
                    # 在这个episode下计算到目标的距离
                    initial_distance = \
                        sldist(
                            [positions[start_index].location.x, positions[start_index].location.y],
                            [positions[end_index].location.x, positions[end_index].location.y])

                    time_out = experiment_suite.calculate_time_out(
                        self._get_shortest_path(positions[start_index], positions[end_index]))

                    # running the agent
                    # 运行 agent 跳转到_run-navigation_episode
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    (result, reward_vec, control_vec, final_time, remaining_distance,steps) = \
                        self._run_navigation_episode(
                            agent, client, time_out, positions[end_index],
                            str(experiment.Conditions.WeatherId) + '_'
                            + str(experiment.task) + '_' + str(start_index)
                            + '.' + str(end_index), RL, total_steps,rl_episode)
                    total_steps = steps
                    if total_steps%1000 == 0:
                        RL.save()
                        print("保存参数"+total_steps)
                    # Write the general status of the just ran episode
                    self._recording.write_summary_results(
                        experiment, pose, rep, initial_distance,
                        remaining_distance, final_time, time_out, result)

                    # Write the details of this episode.
                    self._recording.write_measurements_results(experiment, rep, pose, reward_vec,
                                                               control_vec)
                    if result > 0:
                        logging.info('+++++ Target achieved in %f seconds! +++++',
                                     final_time)
                    else:
                        logging.info('----- Timeout! -----')

            start_pose = 0

        self._recording.log_end()

        return metrics_object.compute(self._recording.path)

    def get_path(self):
        """
        Returns the path were the log was saved.
        """
        return self._recording.path

    def _get_directions(self, current_point, end_point):
        """
        Class that should return the directions to reach a certain goal
        """

        directions = self._planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.orientation.x,
             current_point.orientation.y,
             current_point.orientation.z),
            (end_point.location.x, end_point.location.y, 0.22),
            (end_point.orientation.x, end_point.orientation.y, end_point.orientation.z))
        return directions

    def _get_shortest_path(self, start_point, end_point):
        """
        Calculates the shortest path between two points considering the road netowrk
        """

        return self._planner.get_shortest_path_distance(
            [
                start_point.location.x, start_point.location.y, 0.22], [
                start_point.orientation.x, start_point.orientation.y, 0.22], [
                end_point.location.x, end_point.location.y, end_point.location.z], [
                end_point.orientation.x, end_point.orientation.y, end_point.orientation.z])

    def _run_navigation_episode(
            self,
            agent,
            client,
            time_out,
            target,
            episode_name,rl,total_steps,episode):
        """
         Run one episode of the benchmark (Pose) for a certain agent.
         这是运行一次episode


        Args:
            agent: the agent object
            client: an object of the carla client to communicate
            with the CARLA simulator
            time_out: the time limit to complete this episode
            target: the target to reach
            episode_name: The name for saving images of this episode

        """

        # Send an initial command.
        # 发送一个初始控制
        measurements, sensor_data = client.read_data()
        client.send_control(VehicleControl())

        initial_timestamp = measurements.game_timestamp
        current_timestamp = initial_timestamp

        # The vector containing all measurements produced on this episode
        # 包含此剧集中生成的所有测量值的向量
        measurement_vec = []
        # The vector containing all controls produced on this episode
        # 包含所有控制
        control_vec = []
        frame = 0
        distance = 10000
        success = False

        #while (current_timestamp - initial_timestamp) < (time_out * 1000) and not success:
        while True and not success:

            # Read data from server with the client
            # 从客户端获取到数据
            measurements, sensor_data = client.read_data()
            # The directions to reach the goal are calculated.
            # 计算到达目标的方向
            directions = self._get_directions(measurements.player_measurements.transform, target)
            # Agent process the data.
            # 从agent返回动作和observation,然后在发送动作之后获取下一个observation_
            control, observation, action= agent.run_step(measurements, sensor_data, directions, target,rl)
            # Send the control commands to the vehicle
            # 将控制命令发送到车辆
            client.send_control(control)
            image_RGB = to_rgb_array(sensor_data['CameraRGB'])
            image_RGB_real = image_RGB.flatten()
            observation_ = image_RGB_real

            # save images if the flag is activated
            # 如果激活标记，则保存图像
            self._recording.save_images(sensor_data, episode_name, frame)

            current_x = measurements.player_measurements.transform.location.x
            current_y = measurements.player_measurements.transform.location.y

            logging.info("Controller is Inputting:")
            logging.info('Steer = %f Throttle = %f Brake = %f ',
                         control.steer, control.throttle, control.brake)

            current_timestamp = measurements.game_timestamp
            # Get the distance travelled until now
            # 获取到目前为止的距离    也就是reward
            distance = sldist([current_x, current_y],
                              [target.location.x, target.location.y])
            dista = sldist([current_x, current_y],
                              [395.95, 308.2])

            player_measurements = measurements.player_measurements
            other_lane = 100 * player_measurements.intersection_otherlane
            offroad = 100 * player_measurements.intersection_offroad
            reward = dista
            ### RLstore
            rl.store_transition(observation,action,reward,observation_)
            if total_steps > 100:
                rl.learn()
            total_steps += 1
            # Write status of the run on verbose mode
            # 在详细模式下写入运行状态
            logging.info('Status:')
            logging.info(
                '[d=%f] c_x = %f, c_y = %f ---> t_x = %f, t_y = %f',
                float(distance), current_x, current_y, target.location.x,
                target.location.y)
            # Check if reach the target
            # 检查是否到达目标
            if distance < self._distance_for_success:
                success = True

            # Increment the vectors and append the measurements and controls.
            # 增加矢量并附加测量和控制。
            frame += 1
            measurement_vec.append(measurements.player_measurements)
            control_vec.append(control)
            col = player_measurements.collision_other
            if offroad > 10 or other_lane > 10 or col > 0:
                print('终止条件触发')
                print('episode: ', episode,
                      'total_steps', total_steps,
                      'ep_r: ', round(reward, 2),
                      ' epsilon: ', round(rl.epsilon, 2))
                return 0, measurement_vec, control_vec, time_out, distance, total_steps


            if success:
                return 1, measurement_vec, control_vec, float(
                    current_timestamp - initial_timestamp) / 1000.0, distance, total_steps
        return 0, measurement_vec, control_vec, time_out, distance, total_steps


def run_driving_benchmark(agent,
                          experiment_suite,
                          city_name='Town01',
                          log_name='Test',
                          continue_experiment=False,
                          host='127.0.0.1',
                          port=2000
                          ):
    while True:
        try:

            with make_carla_client(host, port) as client:
                # Hack to fix for the issue 310, we force a reset, so it does not get
                #  the positions on first server reset.
                client.load_settings(CarlaSettings())
                client.start_episode(0)

                # We instantiate the driving benchmark, that is the engine used to
                # benchmark an agent. The instantiation starts the log process, sets
                # 我们实例化驾驶基准，即用于对代理进行基准测试的引擎。 实例化启动日志过程，设置

                benchmark = DrivingBenchmark(city_name=city_name,
                                             name_to_save=log_name + '_'
                                                          + type(experiment_suite).__name__
                                                          + '_' + city_name,
                                             continue_experiment=continue_experiment)
                # This function performs the benchmark. It returns a dictionary summarizing
                # the entire execution.
                # 此功能执行基准测试。 它返回一个总结整个执行的字典。

                benchmark_summary = benchmark.benchmark_agent(experiment_suite, agent, client)

                print("")
                print("")
                print("----- Printing results for training weathers (Seen in Training) -----")
                print("")
                print("")
                results_printer.print_summary(benchmark_summary, experiment_suite.train_weathers,
                                              benchmark.get_path())

                print("")
                print("")
                print("----- Printing results for test weathers (Unseen in Training) -----")
                print("")
                print("")

                results_printer.print_summary(benchmark_summary, experiment_suite.test_weathers,
                                               benchmark.get_path())

                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

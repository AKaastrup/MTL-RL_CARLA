import atexit
import os
import signal
import sys
import subprocess
import carla
import gym
import time
import random
import cv2
import numpy as np
import math
from queue import Queue
from gym import spaces
from absl import logging
import pygame

logging.set_verbosity(logging.INFO)

class CarlaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self,
                obs_space : str,
                start_location : str = 'highway',
                view : bool = False
                ):

        """Initiates the custom gym.Env environment implemented with the CARLA simulator. (Developed with CARLA 0.9.11).
        Args:
            obs_space: What observation space, and thereby which sensors, are used for the simulation
                       Accepted inputs: ['rgb', 'CnnMtl', 'MipMtl']
            start_location: Start location of vehicle upon starting of each training episode.
                            Accepted inputs: ['random', 'highway']
            view: If the new process running CARLA should open a viewable window.
        """

        assert obs_space in ("rgb", "CnnMtl", "MipMtl")
        assert start_location in ("random", "highway")

        self.obs_space = obs_space
        self.start_location = start_location
        self.view = view

        self.client, self.world, self.frame, self.server = self._setup()

        self.client.set_timeout(5.0)
        self.map = self.world.get_map()
        self.spectator = self.world.get_spectator()
        blueprint_library = self.world.get_blueprint_library()
        self.lincoln = blueprint_library.filter('lincoln')[0]

        self.im_width = 100
        self.im_height = 100
        self.repeat_action = 4 #changed from 4 to 25 the 31 may tp accomidate increase in frame rate (not immediately, one trained with 6000 step but 4 repreat)
        self.action_type = 'continuous'
        self.steps_per_episode = 6000 # changed from 600 to 6000 the 31 may to accomidate the increase in frame rate
        self.actor_list = []


        # Create window and placeholder array to save camara images into. Will display input to user
        # Only viable in -opengl envionment. Current computer does not have sufficient GPU to open a viewable CARLA simulatioin with -opengl.
        # if not self.view:
        #     if self.obs_space == 'rgb':
        #         self.obs_vis=np.zeros((self.im_height, self.im_width,3))
        #     else:
        #         self.obs_vis=np.zeros((self.im_height, self.im_width*3,3))
        #     cv2.namedWindow('Observation visualisation',1)


    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor(s)."""
        #Baseline rgb only implementation
        if self.obs_space == 'rgb':
            return gym.spaces.Box(low=0.0, high=255.0, shape=(self.im_height, self.im_width, 3), dtype=np.uint8)

        #Proposed combined network using singular stacked input for rgb semantic segmentation and depth regression to simulate combined MTL-RL network
        #Intended for CnnPolicy model
        elif self.obs_space == 'CnnMtl':
            return gym.spaces.Box(low=0.0, high=255.0, shape=(self.im_height, self.im_width, 7), dtype=np.uint8)

        #Proposed combined network using seperate inputs for rgb semantic segmentation and depth regression to simulate combined MTL-RL network
        #Intended for MultiInputPolicy model
        elif self.obs_space == 'MipMtl':
            spaces = {
                'rgb': gym.spaces.Box(low=0.0, high=255.0, shape=(self.im_height, self.im_width, 3), dtype=np.uint8),
                'semantic_segmentation': gym.spaces.Box(low=0.0, high=255.0, shape=(self.im_height, self.im_width, 3), dtype=np.uint8),
                'depth': gym.spaces.Box(low=0.0, high=255.0, shape=(self.im_height, self.im_width, 3), dtype=np.uint8)}
            dict_space = gym.spaces.Dict(spaces)
            return dict_space

        else:
            raise NotImplementedError()

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        if self.action_type == 'continuous': #Currently the action_type is hard coded as continues
            return gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
        elif self.action_type == 'discrete': #Implemented to anticipate training with discrete systems
            return gym.spaces.MultiDiscrete([4, 9])
        else:
            raise NotImplementedError()

    # Seed used for what?
    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed)
        return seed

    # Resets environment for new episode
    def reset(self):
        # Car, sensors, etc will be recreated after each crash.
        self._destroy_agents()

        logging.debug("Resetting environment")

        # Reset variables for data collection via CARLA API
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.actor_list = []
        self.frame_step = 0
        self.out_of_loop = 0
        self.dist_from_start = 0

        # Define required sensor types from CARLA
        if self.obs_space == 'rgb':
            self.sensors = ['rgb']
        elif self.obs_space == 'CnnMtl' or self.obs_space == 'MipMtl':
            self.sensors = ['rgb', 'semantic_segmentation', 'depth']

        # Set up queues for storing CARLA sensor outputs
        self.camara_queues = dict()
        for key in self.sensors:
            self.camara_queues[key] = Queue()

        # When Carla breaks (stops working) or spawn point is already occupied, spawning a car throws an exception
        # We allow it to try for 3 seconds then forgive
        spawn_start = time.time()
        while True:
            try:
                #Spawn vehicle
                self.start_transform = self._get_start_transform()
                self.curr_loc = self.start_transform.location
                self.vehicle = self.world.spawn_actor(self.lincoln, self.start_transform)
                break
            except Exception as e:
                logging.error('Error carla 141 {}'.format(str(e)))
                time.sleep(0.01)
            # If that can't be done in 3 seconds - forgive (and allow main process to handle for this problem)
            if time.time() > spawn_start + 3:
                raise Exception('Can\'t spawn a car')

        # Append actor to a list of spawned actors to facilitate destorying them for next reset
        self.actor_list.append(self.vehicle)

        # Collect blueprints for needed camaras from CARLA
        self.cams = dict()
        for key in self.sensors:
            self.cams[key] = self.world.get_blueprint_library().find(f'sensor.camera.{key}')

        # Set camera attribute for each camaras
        for key in self.cams:
            #print(type(self.cams[key]))
            self.cams[key].set_attribute('image_size_x', f'{self.im_width}')
            self.cams[key].set_attribute('image_size_y', f'{self.im_height}')
            self.cams[key].set_attribute('fov', '90')


        # Create fixed points relative to spawned vehicle
        bound_x = self.vehicle.bounding_box.extent.x
        bound_y = self.vehicle.bounding_box.extent.y

        # Create CARLA type transform object for front of vehicle
        transform_front = carla.Transform(carla.Location(x=bound_x, z=1.0))

        # Spawn and activate each camera. Add to actor list to faciliate destorying for next reset
        for key in self.cams:
            self.cams[key] = self.world.spawn_actor(self.cams[key], transform_front, attach_to=self.vehicle)
            self.cams[key].listen(self.camara_queues[key].put)
            self.actor_list.extend([self.cams[key]])

        # Work around for initialising car (Needed?)
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        time.sleep(4)

        # Collision history is a list callback is going to append to (End simulation after a collision)
        self.collision_hist = []
        self.lane_invasion_hist = []

        # Collect blueprint for collision and lane sensors
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        # Spawn and activate sensors
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.vehicle)
        self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
        self.colsensor.listen(self._collision_data)
        self.lanesensor.listen(self._lane_invasion_data)
        # Append to actor list to faciliate deleting at next reset
        self.actor_list.append(self.colsensor)
        self.actor_list.append(self.lanesensor)

        self.world.tick()
        if self.view:
            self._update_spectator()

        # Wait for a camera to send first image (important at the beginning of first episode)
        while self.camara_queues['rgb'].empty():
            logging.debug("waiting for camera to be ready")
            time.sleep(0.01)
            self.world.tick()

        # Disengage brakes
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        #Create observation
        obs = self._create_observation()
        return obs

    # Step function made publically available
    def step(self, action):
        total_reward = 0
        for _ in range(self.repeat_action):
            obs, rew, done, info = self._step(action)
            total_reward += rew
            if done:
                break
        return obs, total_reward, done, info

    # Steps environment
    def _step(self, action):
        self.world.tick()
        self.frame_step += 1 #Used??
        self._update_spectator()

        # Apply control to the vehicle based on an action
        # Model returns a two dimentional action, where the first dimention informs throttle vs breaking and the second faciliates steering
        if self.action_type == 'continuous': #Both dimentions yield value in range [-1,1]
            if action[0] > 0:
                action = carla.VehicleControl(throttle=float(action[0]), steer=float(action[1]), brake=0)
            else:
                action = carla.VehicleControl(throttle=0, steer=float(action[1]), brake= -float(action[0]))
        elif self.action_type == 'discrete':
            if action[0] == 0:
                action = carla.VehicleControl(throttle=0, steer=float((action[1] - 4)/4), brake=1)
            else:
                action = carla.VehicleControl(throttle=float((action[0])/3), steer=float((action[1] - 4)/4), brake=0)
        else:
            raise NotImplementedError()
        logging.debug('{}, {}, {}'.format(action.throttle, action.steer, action.brake))
        self.vehicle.apply_control(action)

        if self.view:
            self._update_spectator()


        # Get next observations
        obs = self._create_observation()

        # Calculate distance from start location
        loc = self.vehicle.get_location()
        new_dist_from_start = loc.distance(self.start_transform.location)
        square_dist_diff = new_dist_from_start ** 2 - self.dist_from_start ** 2
        self.dist_from_start = new_dist_from_start

        # Calculate speed in km/h from car's velocity (3D vector)
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        # Reset variables for done, reward and info to be filled out based on following logic
        done = False
        reward = 0
        info = dict()

        # If vehicle collided - end and episode and send back a penalty
        if len(self.collision_hist) != 0:
            done = True
            reward += -100
            self.collision_hist = []
            self.lane_invasion_hist = []

        # If vehicle crossed lane - give penalty
        if len(self.lane_invasion_hist) != 0:
            reward += -5
            self.lane_invasion_hist = []

        # If moving too slow - give proportional penalty
        if kmh < 15:
            reward -= (20-kmh)*0.1
        # If moving at acceptable speed - give proportional reward
        else:
            reward += 0.1 * kmh

        # Give reward for moving away from starting point
        reward += square_dist_diff

        # End episode if episode continued for max allowed length (avoided crash)
        if self.frame_step >= self.steps_per_episode:
            done = True

        # End episode if intended to be on highway, but has exited
        if not self._on_highway():
            self.out_of_loop += 1
            if self.out_of_loop >= 20:
                done = True
        else:
            self.out_of_loop = 0

        if done:
            logging.debug("Env lasts {} steps, restarting ... ".format(self.frame_step))
            self._destroy_agents()
        return obs, reward, done, info

    # Close environment and CARLA simulation process. For usage after completing training.
    def close(self):
        logging.info("Closes the CARLA server with process PID {}".format(self.server.pid))
        os.killpg(self.server.pid, signal.SIGKILL)
        atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))

    # Render functionality from Gymnasium class. Not used for this implementation but is replaced by view functionality to faciliate use of Unreal Engine simulation.
    def render(self, mode='human'):
        if view:
            self._update_spectator()

    # Update CARLA spectator to stay with vehicle
    def _update_spectator(self):
        viewer_transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-4.5, z=2.5)), self.vehicle.get_transform().rotation)
        self.spectator.set_transform(viewer_transform)

    #Collect first images from eah of the used cameras. Process into desired color. Convert to array for manipulation
    def _create_observation(self):
        images = dict()
        for key in self.sensors:
            image = self.camara_queues[key].get()
            if key == 'semantic_segmentation':
                image.convert(carla.ColorConverter.CityScapesPalette)
            if key == 'depth':
                image.convert(carla.ColorConverter.LogarithmicDepth)
            image = np.array(image.raw_data)
            image = image.reshape((self.im_height, self.im_width, -1))
            image = image[:, :, :3]
            #cv2.imwrite(f'{key}.jpeg',image)
            images[key] = image

        # Format image output to match current obseration space
        if self.obs_space == 'rgb':
            obs = images[self.sensors[0]]
        elif self.obs_space == 'CnnMtl':
            obs=np.zeros((self.im_height, self.im_width,7))
            obs[:,:,0:3]=images['rgb']
            obs[:,:,3:4]=images['depth'][:,:,0].reshape(self.im_height, self.im_width, 1)
            obs[:,:,4:7]=images['semantic_segmentation']
        elif self.obs_space == "MipMtl":
            obs = images


        # For view mode save observations in array and show to user
        # Only viable in -opengl envionment. Current computer does not have sufficient GPU to open a viewable CARLA simulatioin with -opengl.
        # if not self.view:
        #     if self.obs_space == 'rgb':
        #         self.obs_vis = obs / 255
        #     else:
        #         self.obs_vis[:,self.im_width*0:self.im_width*1,0:3]=images['rgb'] / 255.0
        #         self.obs_vis[:,self.im_width*1:self.im_width*2,0:3]=images['depth'] / 255.0
        #         self.obs_vis[:,self.im_width*2:self.im_width*3,0:3]=images['semantic_segmentation'] / 255.0
        #
        #     cv2.imshow('Observation visualisation',self.obs_vis)
        #     cv2.waitKey(1)

        return obs

    # Destoys vehicle and attached sensor to facilicate reset after crash
    def _destroy_agents(self):

        for actor in self.actor_list:

            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # If it's still alive - desstroy it
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

    def _collision_data(self, event):

        # What we collided with and what was the impulse
        collision_actor_id = event.other_actor.type_id
        collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        # # Filter collisions
        # for actor_id, impulse in COLLISION_FILTER:
        #     if actor_id in collision_actor_id and (impulse == -1 or collision_impulse <= impulse):
        #         return

        # Add collision
        self.collision_hist.append(event)

    def _lane_invasion_data(self, event):
        # Change this function to filter lane invasions
        self.lane_invasion_hist.append(event)

    def _on_highway(self):
        goal_abs_lane_id = 4
        vehicle_waypoint_closest_to_road = \
            self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        road_id = vehicle_waypoint_closest_to_road.road_id
        lane_id_sign = int(np.sign(vehicle_waypoint_closest_to_road.lane_id))
        assert lane_id_sign in [-1, 1]
        goal_lane_id = goal_abs_lane_id * lane_id_sign
        vehicle_s = vehicle_waypoint_closest_to_road.s
        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
        return not (goal_waypoint is None)

    # Create CARLA transform to inform spawn location of vehicle.
    def _get_start_transform(self):
        if self.start_location == 'random':
            return random.choice(self.map.get_spawn_points())
        if self.start_location == 'highway':
            if self.map.name == "Town04":
                for trial in range(10):
                    start_transform = random.choice(self.map.get_spawn_points())
                    start_waypoint = self.map.get_waypoint(start_transform.location)
                    if start_waypoint.road_id in list(range(35, 50)):
                        break
                return start_transform
            else:
                raise NotImplementedError

    # Initiate a seperate process to run the CARLA simulation
    # Notice: Requires that CARLA implementation is stored at /opt/carla-simulator/
    def _setup(
        self,
        town: str = "Town04",
        fps: int = 100, #changed from 10 to 100 on 30 may
        server_timestop: float = 10.0,
        client_timeout: float = 60.0,
        num_max_restarts: int = 10,
    ):
        """Returns the `CARLA` `server`, `client` and `world`.
        Args:
            view: If the new process running CARLA should open a viewable window.
            town: The `CARLA` town identifier.
            fps: The frequency (in Hz) of the simulation.
            server_timestop: The time interval between spawing the server
            and resuming program.
            client_timeout: The time interval before stopping
            the search for the carla server.
            num_max_restarts: Number of attempts to connect to the server.
        Returns:
            client: The `CARLA` client.
            world: The `CARLA` world.
            frame: The synchronous simulation time step ID.
            server: The `CARLA` server.
        """

        # The attempts counter.
        attempts = 0

        while attempts < num_max_restarts:
            logging.debug("{} out of {} attempts to setup the CARLA simulator".format(
                attempts + 1, num_max_restarts))

            # Random assignment of port.
            port = np.random.randint(2000, 3000)

            #  Configure CARLA server for optional offscreen option.
            env = os.environ.copy()
            if not self.view:
                env["SDL_VIDEODRIVER"] = "offscreen"
            env["SDL_HINT_CUDA_DEVICE"] = "0"
            logging.debug("Inits a CARLA server at port={}".format(port))

            carla_dir = "/opt/carla-simulator/" #Update as needed
            # Start CARLA server
            if self.view:
                server = subprocess.Popen(str(os.path.join(carla_dir, "CarlaUE4.sh")) + f' -carla-rpc-port={port}' + f" -prefernvidia",
                    stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, shell=True)
            else:
                server = subprocess.Popen(f'DISPLAY= ' + str(os.path.join(carla_dir, "CarlaUE4.sh")) + f' -opengl '+ f' -carla-rpc-port={port}' + f" -quality-level=Epic -prefernvidia", stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, shell=True)
            atexit.register(os.killpg, server.pid, signal.SIGKILL)
            time.sleep(server_timestop)

            # Connect client.
            logging.debug("Connects a CARLA client at port={}".format(port))
            try:
                client = carla.Client("localhost", port)  # pylint: disable=no-member
                client.set_timeout(client_timeout)
                client.load_world(map_name=town)
                world = client.get_world()
                world.set_weather(carla.WeatherParameters.ClearNoon)  # pylint: disable=no-member

                # if self.view:
                #     frame = world.apply_settings(
                #         carla.WorldSettings(  # pylint: disable=no-member
                #             fixed_delta_seconds=1.0 / fps
                #         ))
                #else:
                frame = world.apply_settings(
                    carla.WorldSettings(  # pylint: disable=no-member
                        synchronous_mode=True,
                        fixed_delta_seconds= 1.0 / fps))
                logging.debug("Server version: {}".format(client.get_server_version()))
                logging.debug("Client version: {}".format(client.get_client_version()))
                return client, world, frame, server
            except RuntimeError as msg:
                logging.debug(msg)
                attempts += 1
                logging.debug("Stopping CARLA server at port={}".format(port))
                os.killpg(server.pid, signal.SIGKILL)
                atexit.unregister(lambda: os.killpg(server.pid, signal.SIGKILL))

        logging.debug(
            "Failed to connect to CARLA after {} attempts".format(num_max_restarts))
        sys.exit()

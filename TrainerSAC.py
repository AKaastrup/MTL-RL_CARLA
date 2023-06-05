import gym
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from carla_env import CarlaEnv
from datetime import date
import numpy as np
import os

class TrainerSAC():

        def __init__(self,obs_space : str, start_location : str = 'highway'):

            self.obs_space = obs_space
            self.env = CarlaEnv(obs_space, start_location, view=False)
            self.env.reset()

            if obs_space == 'rgb' or obs_space == 'CnnMtl':
                self.model = SAC(
                "CnnPolicy",
                self.env,
                verbose=1,
                seed=7,
                device='cuda',
                tensorboard_log='./LOG_SAC',
                action_noise=NormalActionNoise(mean=np.array([0.3, 0]), sigma=np.array([0.5, 0.1]))
                )
            else:
                self.model = SAC(
                "MultiInputPolicy",
                self.env,
                verbose=1,
                seed=7,
                device='cuda',
                tensorboard_log='./LOG_SAC',
                action_noise=NormalActionNoise(mean=np.array([0.3, 0]), sigma=np.array([0.5, 0.1]))
                )

        def train(self, iterations : int):
            model_name, save_path = self._make_names()

            print('Start learning:')
            for i in range(1,iterations+1):
                print(f'Starting training iteration {i}')
                self.model.learn(
                    total_timesteps=10000,
                    log_interval=1,
                    reset_num_timesteps=False,
                    tb_log_name= model_name,
                    )
                print(f'Training iteration {i} complete, saving model.')
                self.model.save(f'{save_path}/Timestep_{i*10000}')

         def cont_train(self, model_path : str, iterations : int):
            model_name, save_path = self._make_names()
            self.model = SAC.load(model_path, env=self.env)

            print(f'Starting learning - continue from {model_path}')
            for i in range(1,iterations+1):
                print(f'Starting training iteration {i}')
                self.model.learn(
                    total_timesteps=10000,
                    log_interval=1,
                    reset_num_timesteps=False,
                    tb_log_name=model_name,
                    )
                print(f'Training iteration {i} complete, saving model.')
                self.model.save(f'{save_path}/Timestep_{i*10000}')

        # Creates unqiue model name and save path to be used in naming the resultant models.
        def _make_names(self):
            model_name = f'SAC_{self.obs_space}_{date.today()}'

            # Check for unique save path
            unique_name = False
            i = 0
            while not unique_name:
                save_path = f'SAC_models/{model_name}_{i}'
                if os.path.exists(save_path):
                    i += 1
                else:
                    unique_name = True
            return f'{model_name}_{i}', save_path

        # Run evaluation of rewards off-screen. implemented due to GPU limitations not allowing Viewer CARLA server to open with -opengl, which alters input slightly
        def evaluate_reward_offscreen(self, model_path : str, iterations : int):
            max = -10000
            min = 10000
            running_sum = 0.0
            self.model = SAC.load(model_path, env=self.env)
            print('Starting offscreen evaluation runs')
            for i in range(iterations):
                obs = self.env.reset()
                done = False
                ep_reward = 0
                while not done:
                    action, _ = self.model.predict(obs)
                    obs, reward, done, _ = self.env.step(action)
                    ep_reward += reward
                print(f"Episode {i+1} reward: {ep_reward}")
                running_sum += ep_reward
                if ep_reward > max:
                    max = ep_reward
                elif ep_reward < min:
                    min = ep_reward
            mean = running_sum / iterations
            print(f"The maximum reward from {iterations} iterations is: {max}")
            print(f"The minimum reward from {iterations} iterations is: {min}")
            print(f"The average reward from {iterations} iterations is: {mean}")

        def close(self):
            self.env.close()

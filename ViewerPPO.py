import gym
from stable_baselines3 import PPO
from carla_env import CarlaEnv


class ViewerPPO():

        def __init__(self, view_model, obs_space : str, start_location : str = 'highway'):

            self.obs_space = obs_space
            self.env = CarlaEnv(obs_space, start_location, view=True)

            self.model = PPO.load(view_model, env=self.env)

        def view(self, iterations : int):
            for i in range(iterations):
                obs = self.env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs)
                    obs, _, done, _ = self.env.step(action)

        def evaluate_reward(self, iterations : int):
            max = -10000
            min = 10000
            running_sum = 0.0
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

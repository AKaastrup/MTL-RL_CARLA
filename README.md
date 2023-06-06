# MTL-RL_CARLA
Simulated combined MTL-RL network trained in CARLA simulation

## Introduction 
This repocitory holds the code implementation for a pseudo MTL Deep-RL autonomous system trained for autonomous nagigation of a vechicle. The implementation is facilicated by the OpenAI Gymnasium and Stable Baselines3 implementations and the CARLA simulation environment. Credit is given to yanlai00/RL-Carla as their custom CARLA gymnasium enviornment was used as a starting point for this implementation. 

This implementation aims to compare the performance of a pseudo MTL Deep-RL arhitecture (two options provided) and a pure Deep-RL architecture to determine if an expansion of the state space facilicated by the MTL component can yield make helpful features more accessable to the subsequent RL component thereby increasing performance. For this use case, an autonomous vehichle is trained to navigate using RGB inputs in an end-to-end approch. For the MTL variants the MTL component would produce the corresponding segmantic segmentation and depth regression for the input image. The RL component would then train on all three images. 

## Dependancies and setup
The implementation uses [OpenAi Gymnaisum](https://gymnasium.farama.org), [Stable Baselines3](https://stable-baselines.readthedocs.io/en/master/) and the [CARLA simulator](https://carla.org) to facilitate the training. This implementation was tested using the 0.9.11 CARLA version and was installed using the Debian CARLA installation method as is described in [this guide](https://carla.readthedocs.io/en/latest/start_quickstart/). 

For alternative CARLA installation is may be necessay to reconfigure the carla_env.py \_setup() function such that the carla_dir variable refers to the correct directory.

## Commandline parameters 
* --training-algorithm: Training algorithm to be used. \[PPO, SAC\]. Default PPO
* --obs_space: Observation space to be used. \[rgb, CnnMtl, MipMtl\]
  * See arhitecture for each option under ?? 
* --view-model: Path of trained model to be run in view mode. Path given relative to directory location of main.py
  * Relevant comment in Other considerations
* --start-location: The spawn location for each episode. For evaluating, it is recommended to choose same option as trained on. \[random, highway\] 
* --cont_model: Model path for excisting model on which training should continue.
* --iterations: The number of iterations to be done. For training 10000 time steps pr iterations. For view and evaluation, 1 run until termination pr episode.
* --evaluate-reward: Flag for if reward statistics are desired for evaluating excisting model
  * Relevant comment in Other considerations  

If cont_model not None, no view-model allowed and evaluate-reward not chosen.
If view-model specifed no cont_model allowed.
If evaluate-reward a view-model must be specifed.

## Run commands
Train new PPO model using observation space MipMtl, 5x10.000 timesteps with incremental save at aver 10.000 timestep.

```bash
python main.py --obs-space MipMtl
```

Continue training on saved model. Requires continued model uses same observation space as original model, assume MipMtl. 5x10.000 timesteps with incremental save at aver 10.000 timestep.
```bash
python main.py --obs-space MipMtl --cont-model <path from main.py directory>
```

View runs from saved model. Requires specifed observations space matches observation space of model, assume MipMtl. 5 episodes run.
```bash
python main.py --obs-space MipMtl --view-model <path from main.py directory>
```
Evaluate rewards for saved model. Requires specifed observations space matches observation space of model, assume MipMtl. Statstics over 50  episodes run.
```bash
python main.py --obs-space MipMtl --view-model <path from main.py directory> --evaluate-reward --iterations 50
```

## Arhitectures


## Pretraied models 
Pretrained models for each of the three architectures have been made available in the PPO\_models folder. A folder is provided for each training sessions wherein there have been saved iterative versions every 10.000 timesteps. The corresponding timestep for each saved model is given in the title. For convinience the version with the highest episode reward mean as provided by the TensorBoard logs have been duplicated and placed in directly in the PPO\_models folder.

* CnnMtl_long is a continuation of CnnMtl_2
* MtlMip_long is a continuation of MtlMip_1
* rgb_long is a continuation of rgb_1

## Other considerations
Due to hardware constraints the Viewer impelemtations could not run using a CARLA simulation server opened with the -opengl and -quality-level=Epic flags. Thus the camera images produced by the CARLA simulation and passed as inputs to the RL components are inferior in the Viewer implementations compared to the Trainer implementation. Specifically, the segmantic segmentation and depth images get black splotches due to omitting the -opengl flag. This may lead to slightly lower performance in the Viewer mode.
Due to this the evaluate_reward option uses the Trainer component and is performed off-screen for the most accurate reward values. 
If a higher capacity is available it is recommended to update the code such that the on-screen CARLA server uses these flags. Found in carla_env.py \_setup().

Update:
```bash
if self.view:
   server = subprocess.Popen(str(os.path.join(carla_dir, "CarlaUE4.sh")) + f' -carla-rpc-port={port}' + f" -prefernvidia", stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, shell=True)
```
to:
```bash
if self.view:
   server = subprocess.Popen(str(os.path.join(carla_dir, "CarlaUE4.sh")) + f' -carla-rpc-port={port}' + f" -opengl" + f" -quality-     level=Epic" + f" -prefernvidia", stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, shell=True)
```

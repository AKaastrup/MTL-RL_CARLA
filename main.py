import argparse
from TrainerPPO import TrainerPPO
from TrainerSAC import TrainerSAC
from ViewerPPO import ViewerPPO
from ViewerSAC import ViewerSAC

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--training-algorithm', default='PPO', help='Training algorithm to be used. [PPO, SAC]')
    parser.add_argument('--obs-space', help='Observation space to be used. [rgb, CnnMtl, MipMtl]')
    parser.add_argument('--view-model', default=None, type=str, help='Path of trained model to be run in view mode. \
                        Path given relative to directory location of main.py')
    parser.add_argument('--start-location', default='highway', help='The spawn location for each episode. \
                        For viewing, it is recommended to choose same option as trained on. [random, highway]')
    parser.add_argument('--iterations', default=5, type=int, help='The number of training or view iterations to be done. \
                        10000 timesteps pr training iteration where each iteration continues from previous trained model. \
                        1 episode run pr iteration for viewing and evaluating.')
    parser.add_argument('--continue-model', default=None, type=str, help='Path of an existing model where\
                        further training should be made using the current policy as a starting point')
    parser.add_argument('--evaluate-reward', action='store_true', help='Will print statistics of rewards for \
                        the run episode of provided model.')

    args = parser.parse_args()
    algorithm = args.training_algorithm.upper()
    obs_space = args.obs_space
    view_model = args.view_model
    start_location = args.start_location
    iterations = args.iterations
    cont_model = args.continue_model
    evaluate_reward = args.evaluate_reward

    assert algorithm in ("PPO", "SAC")
    assert obs_space in ("rgb", "CnnMtl", "MipMtl")
    assert start_location in ("random", "highway")
    if cont_model != None:
        assert view_model == None
    if evaluate_reward:
        assert view_model != None


    # For this implementation evaluate_reward will be done off screen due to not being able to use -opengl (carla_env.py) when opening onscreen server
    if view_model == None or evaluate_reward:
        if algorithm == "PPO":
            trainer = TrainerPPO(obs_space, start_location)
        else:
            trainer = TrainerSAC(obs_space, start_location)
        if cont_model != None:
            trainer.cont_train(cont_model, iterations)
        elif evaluate_reward:
            trainer.evaluate_reward_offscreen(view_model, iterations)
        else:
            trainer.train(iterations)
        trainer.close()
    else:
        if algorithm == "PPO":
            viewer = ViewerPPO(view_model, obs_space, start_location)
        else:
            viewer = ViewerSAC(view_model, obs_space, start_location)
        viewer.view(iterations)
        viewer.close()

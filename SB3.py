# RL Libraries
import gym
from stable_baselines3 import PPO

# Utility Libraries
from tqdm import tqdm

# Internal Libraries
import os

# OOP the Code 
# TODO: Reward Function Programs, Model Analysis Dashboard, Model Comparison Function. 

def environement_setup(env_name):
    '''Creates the Enviroment by passing an environment string'''
    env = gym.make(env_name)
    env.reset()
    return env


def create_model_directory(algorithm):
    '''Creates Model Directory
    Algorithm is the external librarys model(stable_baselines3)'''
    models_dir = "models"+"/"+algorithm.__class__.__name__
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return models_dir

def declaring_model(algorithm, env, verbose=1):
    '''Declares the model'''
    model = algorithm(
        'MlpPolicy',
        env,
        verbose=1,
    )
    return model

def model_learning(model, time_steps):
    model.learn(total_timesteps=time_steps)

if __name__ == "__main__":
    TIME_STEPS = 10000

    iteration = 0
    episodes = 30

    training = False
    deployment = True

    algorithm = PPO
    env_name = "LunarLander-v2"

    env = environement_setup(env_name)
    models_dir = create_model_directory(algorithm)
    model = declaring_model(algorithm, env)
    # model_learning(model, TIME_STEPS)

    if training:
        while True:
            iteration += 1
            model.learn(total_timesteps=TIME_STEPS, reset_num_timesteps=False)
            model.save(f"{models_dir}/{TIME_STEPS*iteration}")

    if deployment:
        model_dir = os.getcwd() + os.sep + "models" + os.sep + "ABCMeta"
        model_folder = os.listdir(model_dir)[-1]
        model_path = model_dir + os.sep + model_folder
        print(model_path)
        
        model = PPO.load(model_path, env=env)

        episodes = 1

        for ep in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                action, states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()
                print(rewards)
        env.close()


# This is the driver file

# Credit for environment wrappers, model loading, tryint(), and alpha_num_keys()
# goes to https://github.com/lixado/PyBoy-RL/blob/main/main.py 
# and https://github.com/lixado/PyBoy-RL/blob/main/functions.py
# Credit for training mode and evaluate mode
# goes to https://github.com/lixado/PyBoy-RL/blob/main/main.py 

from pyboy.pyboy import *
import gym
from Kirby.envs.kirby_env import KirbyEnv
import os
from collections import deque
import time
import sys
from gym.wrappers import FrameStack, NormalizeObservation
from Kirby.wrappers import SkipFrame, ResizeObservation
import datetime
from pathlib import Path
from MetricLogger import MetricLogger
import re

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

print("[1] Free Play")
print("[2] Sample Computer Control")
print("[3] Training")
print("[4] Evaluation")
print("[5] Help")
mode = int(input("Select a mode: "))

pyboy = PyBoy('Kirby/ROMs/KirbysDreamLand.gb', window_type="SDL2", window_scale=3, debug=False, game_wrapper=True)

# Explicitly set the game speed to normal
pyboy.set_emulation_speed(1)
assert pyboy.cartridge_title() == "KIRBY DREAM LA"
# The game wrapper allows access to in-game variables
kirby = pyboy.game_wrapper()

# Paths for models
now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir = Path("checkpoints") / now
save_dir_eval = Path("checkpoints") / (now + "-eval")
checkpoint_dir = Path("checkpoints")

frameStack = 4
gameDimensions = (20, 16)
episodes = 150
# In seconds
timeLimit = 60

# Initialize environment
kirb_env = KirbyEnv(pyboy, (frameStack,) + gameDimensions, observation_type="tiles", save_dir=save_dir)

# Apply wrappers
kirb_env = SkipFrame(kirb_env, skip=4)
kirb_env = ResizeObservation(kirb_env, gameDimensions)  # transform MultiDiscreate to Box for framestack
kirb_env = NormalizeObservation(kirb_env)  # normalize the values
kirb_env = FrameStack(kirb_env, num_stack=frameStack)
# Due to all the wrappers, it's no longer possible to access environment variables/properties directly
# But it's still possible to call the environment's defined functions/methods
# Getter functions should be used to access environment variables instead

if mode == 1:
    kirby.start_game()
    while True:
        pyboy.tick()
elif mode == 2:
    kirby.start_game()

    assert kirby.score == 0
    assert kirby.lives_left == 4
    assert kirby.health == 6

    # This stays pressed
    pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)

    # Walk for 280 frames
    for _ in range(280):
        pyboy.tick()

    assert kirby.score == 800
    assert kirby.health == 5

    kirb_env.reset()
    assert kirby.score == 0
    assert kirby.health == 6

    kirb_env.close()
    exit()
elif mode == 3:
    print("Training mode selected")

    save_dir.mkdir(parents=True)
    logger = MetricLogger(save_dir)

    print(f"Number of episodes: {episodes}")

    kirby.start_game()
    kirb_env.net.train()
    for episode in range(episodes):
        # Ticking doesn't start until past the title screen and level intro animation
        print("Starting a new episode")
        observation, info = kirb_env.reset()
        assert kirby.score == 0
        assert kirby.lives_left == 4
        assert kirby.health == 6
        start_time = time.time()
        while True:
            action_space_Indx = kirb_env.act(observation)
            next_observation, reward, done, truncated, info = kirb_env.step(action_space_Indx)
            # Remember
            kirb_env.cache(observation, next_observation, action_space_Indx, reward, done)
            # Learn
            q, loss = kirb_env.learn()
            # Logging
            logger.log_step(reward, loss, q, kirb_env.scheduler.get_last_lr())
            # Update state
            observation = next_observation

            if done or (time.time() - start_time >= timeLimit):
                print("Termination condition triggered, ending episode")
                kirb_env.reset()
                break

        logger.log_episode()
        logger.record(episode=episode, epsilon=kirb_env.get_Epsilon(), stepsThisEpisode=kirb_env.get_CurrentStep(), maxLength=kirb_env.get_Length())

    kirb_env.save()
    kirb_env.close()
elif mode == 4:
    print("Evaluate mode selected")

    # Load a previous model
    # List of checkpoint folders names
    folderList = [name for name in os.listdir(checkpoint_dir) if os.path.isdir(checkpoint_dir / name) and len(os.listdir(checkpoint_dir / name)) != 0]

    if len(folderList) == 0:
        print("No models to load in path: ", save_dir)
        quit()

    for index, fileName in enumerate(folderList, 1):
        sys.stdout.write("[%d] %s\n\r" % (index, fileName))

    choice = int(input("Select folder with platformer model[1-%s]: " % index)) - 1
    folder = folderList[choice]
    print(f"{folder} selected")

    fileList = [f for f in os.listdir(checkpoint_dir / folder) if f.endswith(".chkpt")]
    # Ensure models are sorted by their numbers
    fileList.sort(key=alphanum_key)
    if len(fileList) == 0:
        print("No models to load in path: ", folder)
        quit()

    # Get the most recent model
    modelPath = checkpoint_dir / folder / fileList[-1]
    kirb_env.loadModel(modelPath)

    save_dir_eval.mkdir(parents=True)
    logger = MetricLogger(save_dir_eval)

    print(f"Number of episodes: {episodes}")

    kirby.start_game()
    kirb_env.net.eval()
    for episode in range(episodes):
        # Ticking doesn't start until past the title screen and level intro animation
        print("Starting a new episode")
        observation, info = kirb_env.reset()
        assert kirby.score == 0
        assert kirby.lives_left == 4
        assert kirby.health == 6
        while True:
            action_space_Indx = kirb_env.act(observation)
            next_observation, reward, done, truncated, info = kirb_env.step(action_space_Indx)
            # Logging
            logger.log_step(reward, 1, 1, 1)
            # Update state
            observation = next_observation

            if done:
                print("Termination condition triggered, ending episode")
                kirb_env.reset()
                break

        logger.log_episode()
        logger.record(episode=episode, epsilon=kirb_env.get_Epsilon(), stepsThisEpisode=kirb_env.get_CurrentStep(), maxLength=kirb_env.get_Length())
    
    kirb_env.close()
elif mode == 5:
    print("Free Play - Take control of Kirby yourself as you play through Kirby's Dream Land")
    print("Sample Computer Control - Watch an example of how the computer can control Kirby")
    print("Training - Train the agent")
    print("Evaluation - Watch the agent act based on training")
    exit()
else:
    print("Invalid selection, ending program")
    exit()
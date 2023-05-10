# Credit for setting up the DDQN, act(), learn(), and all functions related to the latter
# goes to https://github.com/lixado/PyBoy-RL/blob/main/agent.py 
# Credit for step() 
# goes to https://github.com/lixado/PyBoy-RL/blob/main/CustomPyBoyGym.py
# Inspiration for the GameState class, reward(), and initializeActions()
# goes to https://github.com/lixado/PyBoy-RL/blob/main/AISettings/KirbyAISettings.py 

import numpy as np
import gym
from pyboy.pyboy import PyBoyGymEnv, WindowEvent
import itertools
import torch
# Remember that imports occur from the project's root directory, not the file's relative location
from Kirby.learning_model import DDQN
from collections import deque
import random

class GameState:
  def __init__(self, pyboy):
    game_wrapper = pyboy.game_wrapper()
    self.screen_x_position = pyboy.get_memory_value(0xD053)
    self.kirby_x_position = pyboy.get_memory_value(0xD05C)
    self.kirby_y_position = pyboy.get_memory_value(0xD05D)
    self.game_state = pyboy.get_memory_value(0xD02C)
    self.health = game_wrapper.health
    self.score = game_wrapper.score

class KirbyEnv(PyBoyGymEnv): 
  metadata = {'render_modes': ['human']}

  def __init__(self, pyboy, state_dimensions, observation_type, save_dir, render_mode=None):
    self.pyboy = pyboy
    self.game_wrapper = pyboy.game_wrapper()
    
    self.initializeActions()

    self.state_dimensions = state_dimensions
    self.observation_type = observation_type
    self.save_dir = save_dir

    self.device = "cpu"
    if torch.cuda.is_available():
        self.device = "cuda"

    self.net = DDQN(self.state_dimensions, self.action_space_length).to(device=self.device)

    self.curr_step = 0

    # Exploration rate
    self.epsilon = 1
    # self.epsilon_decay = 0.9999975
    self.epsilon_decay = 0.999975
    self.epsilon_min = 0.01

    # For memory
    self.deque_size = 500000
    self.memory = deque(maxlen=self.deque_size)
    self.batch_size = 64
    self.save_every = 2e5

    # For Q learning
    self.gamma = 0.8
    self.learning_rate = 0.0002
    self.learning_rate_decay = 0.9999985
    self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
    self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.learning_rate_decay)
    self.loss_fn = torch.nn.SmoothL1Loss()
    self.burnin = 1000
    self.learn_every = 3
    self.sync_every = 100
  
  ######
  # initializeActions()
  ######
  def initializeActions(self):
    """Provides the action space."""
    # Using the press window event
    buttons = [WindowEvent.PRESS_ARROW_UP,
              WindowEvent.PRESS_ARROW_DOWN,
              WindowEvent.PRESS_ARROW_RIGHT,
              WindowEvent.PRESS_ARROW_LEFT,
              WindowEvent.PRESS_BUTTON_A,
              WindowEvent.PRESS_BUTTON_B]
    # This list is [0, 1, 2, 3, 4, 5]
    self.buttons = buttons
    # Using the release window event
    release_buttons = [WindowEvent.RELEASE_ARROW_UP,
                      WindowEvent.RELEASE_ARROW_DOWN,
                      WindowEvent.RELEASE_ARROW_RIGHT,
                      WindowEvent.RELEASE_ARROW_LEFT,
                      WindowEvent.RELEASE_BUTTON_A,
                      WindowEvent.RELEASE_BUTTON_B]
    
    self.release_buttons = release_buttons
    # The two lists above contain different elements
    
    self.button_is_pressed = {1 : False,
                              2 : False,
                              3 : False,
                              4 : False,
                              5 : False,
                              6 : False}

    permutations = list(itertools.permutations(buttons, 2))
    combinations = []
    for permutation in permutations:
      reverse_order = permutation[::-1]
      if reverse_order not in combinations:
        combinations.append(permutation)

    # 21 unique action combinations
    self.action_space = [[button] for button in buttons] + combinations
    self.action_space_length = len(self.action_space)

  ######
  # get_Gamestate()
  ######
  def get_Gamestate(self): return GameState(self.pyboy)
  
  ######
  # get_Epsilon()
  ######
  def get_Epsilon(self): return self.epsilon

  ######
  # set_Epsilon()
  ######
  def set_Epsilon(self, epsilon): self.epsilon = epsilon

  ######
  # get_CurrentStep()
  ######
  def get_CurrentStep(self): return self.curr_step

  ######
  # act()
  ######
  def act(self, state):
    # For exploring
    if np.random.rand() <= self.epsilon:
      # actionIdx = np.random.choice(len(self.action_space))
      actionIdx = random.randint(0, self.action_space_length-1)
    # For exploiting
    else:
      state = np.array(state)
      state = torch.tensor(state).float().to(device=self.device)
      state = state.unsqueeze(0)
      
      neuralNetOutput = self.net(state, model="online")
      actionIdx = torch.argmax(neuralNetOutput, axis=1).item()

    # Decrease exploration rate
    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon_min, self.epsilon)

    # Increment step
    self.curr_step += 1

    return actionIdx
  
  ######
  # step()
  ######
  def step(self, action_space_Indx):
    prev_Gamestate = self.get_Gamestate()

    # Get the list of actions to perform from the action space
    action_list = self.action_space[action_space_Indx]

    # Release all buttons previously pressed but not currently needed
    for button, press_status in self.button_is_pressed.items():
      if press_status is True and button not in action_list:
        un_action = self.release_buttons[button-1]
        self.pyboy.send_input(un_action)
        press_status = False

    # Press corresponding buttons for action
    for action_Indx in action_list:
      # action_Indx is an integer from 1-6
      # Get the WindowEvent from self.buttons
      action = self.buttons[action_Indx-1]
      self.pyboy.send_input(action)
      self.button_is_pressed[action_Indx] = True

    # Advance the game by 1 frame
    # Must occur within step() and not in main() so there are two different game states to observe for reward()
    self.pyboy.tick()

    observation = self._get_observation()
    reward = self.get_reward(prev_Gamestate)
    done = self.check_game_over()
    info = {}
    truncated = None

    return observation, reward, done, truncated, info
  
  ######
  # get_reward()
  ######
  def get_reward(self, prev_Gamestate: GameState):
    curr_Gamestate = GameState(self.pyboy)

    if curr_Gamestate.health < prev_Gamestate.health:
      return -200

    # Massive negative reward if dead
    if curr_Gamestate.health == 0 and prev_Gamestate.health != 0:
      return -1000000
    
    # Massive negative reward if Kirby reaches the warpstar
    if curr_Gamestate.health > 0 and curr_Gamestate.game_state == 6 and prev_Gamestate.game_state != 6:
      return -1000000

    if curr_Gamestate.score > prev_Gamestate.score:
      return 100000
    
    # Moving right
    if curr_Gamestate.kirby_x_position > prev_Gamestate.kirby_x_position:
      return 3
    
    # Moving left
    if curr_Gamestate.kirby_x_position < prev_Gamestate.kirby_x_position:
      return 1
    
    # "Standing still"
    if curr_Gamestate.kirby_x_position == prev_Gamestate.kirby_x_position:
      return -30
    
    # Stay away from the level ceiling, no enemies are there
    if curr_Gamestate.kirby_y_position == 16:
      return -1000000
        
    return 0

  ######
  # cache()
  ######
  def cache(self, state, next_state, action, reward, done):
    """
    Store the experience to self.memory (replay buffer)

    Inputs:
    state (LazyFrame),
    next_state (LazyFrame),
    action (int),
    reward (float),
    done(bool))
    """
    state = np.array(state)
    next_state = np.array(next_state)

    state = torch.tensor(state).float().to(device=self.device)
    next_state = torch.tensor(next_state).float().to(device=self.device)
    action = torch.tensor([action]).to(device=self.device)
    reward = torch.tensor([reward]).to(device=self.device)
    done = torch.tensor([done]).to(device=self.device)

    self.memory.append((state, next_state, action, reward, done))

  ######
  # loadModel()
  ######
  def loadModel(self, path):
    dt = torch.load(path, map_location=torch.device(self.device))
    self.net.load_state_dict(dt["model"])
    self.epsilon = dt["epsilon"]
    print(f"Loading model at {path} with exploration rate {self.epsilon}")

  ######
  # learn()
  ######
  def learn(self):
    """Update online action value (Q) function with a batch of experiences"""
    if self.curr_step % self.sync_every == 0:
        self.sync_Q_target()

    if self.curr_step % self.save_every == 0:
        self.save()

    if self.curr_step < self.burnin:
        return None, None

    if self.curr_step % self.learn_every != 0:
        return None, None

    # Sample from memory get self.batch_size number of memories
    state, next_state, action, reward, done = self.recall()

    # Get TD Estimate, make predictions for the each memory
    td_est = self.td_estimate(state, action)

    # Get TD Target make predictions for next state of each memory
    td_tgt = self.td_target(reward, next_state, done)

    # Backpropagate loss through Q_online
    loss = self.update_Q_online(td_est, td_tgt)

    return (td_est.mean().item(), loss)

  ######
  # sync_Q_target()
  ######
  def sync_Q_target(self):
    self.net.target.load_state_dict(self.net.online.state_dict())

  ######
  # save()
  ######
  def save(self):
    """
    Save the state to directory
    """
    save_path = (self.save_dir / f"kirby_net_0{int(self.curr_step // self.save_every)}.chkpt")
    torch.save(
        dict(model=self.net.state_dict(), epsilon=self.epsilon),
        save_path,
    )
    print(f"KirbyNet saved to {save_path} at step {self.curr_step}")

  ######
  # recall()
  ######
  def recall(self):
    """
    Retrieve a batch of experiences from memory
    """
    batch = random.sample(self.memory, self.batch_size)
    state, next_state, action, reward, done = map(torch.stack, zip(*batch))
    return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
  
  ######
  # td_estimate()
  ######
  def td_estimate(self, state, action):
    """
    Output is batch_size number of rewards = Q_online(s,a) * 32
    """
    modelOutPut = self.net(state, model="online")
    current_Q = modelOutPut[np.arange(0, self.batch_size), action]  # Q_online(s,a)
    return current_Q

  ######
  # td_target()
  ######
  @torch.no_grad()
  def td_target(self, reward, next_state, done):
    """
    Output is batch_size number of Q*(s,a) = r + (1-done) * gamma * Q_target(s', argmax_a'( Q_online(s',a') ) )
    """
    next_state_Q = self.net(next_state, model="online") 
    best_action = torch.argmax(next_state_Q, axis=1) # argmax_a'( Q_online(s',a') ) 
    next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action] # Q_target(s', argmax_a'( Q_online(s',a') ) )
    return (reward + (1 - done.float()) * self.gamma * next_Q).float() # Q*(s,a)

  ######
  # update_Q_online()
  ######
  def update_Q_online(self, td_estimate, td_target):
    loss = self.loss_fn(td_estimate, td_target)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.scheduler.step() #

    return loss.item()

  ######
  # get_Length()
  ######
  def get_Length(self): return self.get_Gamestate().health

  ######
  # check_game_over()
  ######
  def check_game_over(self):
    current_kirby = GameState(self.pyboy)
    return True if current_kirby.health <= 0 else False

  ######
  # reset()
  ######
  def reset(self):
    self.game_wrapper.reset_game()

    # Release all buttons currently pressed
    for button, press_status in self.button_is_pressed.items():
      # button is a tuple
      if press_status is True:
        un_action = self.release_buttons[button-1]
        self.pyboy.send_input(un_action)
        press_status = False

    observation = self._get_observation()
    info = {}

    return observation, info

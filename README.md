# KirbyRL
A reinforcement learning project playing Kirby's Dream Land

Inspired by https://github.com/lixado/PyBoy-RL

Recommended Python version: 3.10.6
## How to set up the environment

1. Set up the virtual environment.

- `$ python  -m venv env`

- "env" is the name of the virtual environmnet.

- Use `python3` instead of `python` if there are path issues.

2. Activate the virtual environment.

- On Windows and Command Prompt: `$ env\Scripts\activate.bat`

- On Mac: `$ source env/bin/activate`

3. Install the required modules.

- `$ pip install -e .`

- `$ pip install -r requirements.txt`

## Next Steps:

* [x] Set up the basis for a reinforcement learning environment

* [x] Install gym

* [x] Set up the gym environment class

* [x] Connect the previous file to the driver file

* [x] Bring Kirby game warpper into kirby_env.py

* [ ] ~~Initialize the state space~~

* [x] Initialize the action space

* [ ] ~~Initialize transition table~~

* [x] Initialize epsilon

* [x] Define step()

* [x] Look over the existing project: https://github.com/lixado/PyBoy-RL

* [x] Define the random action part of act()

* [x] Define reset()

* [x] Implement episodes. Two termination conditions: Kirby loses all health or program timer reaches zero

* [x] Move the ROMS folder to inside the Kirby folder so pip doesn't think there are multiple top-level packages

* [x] Make menu

* [x] Implement training mode

* [x] Implement evaluate mode

* [x] Environment wrappers
 

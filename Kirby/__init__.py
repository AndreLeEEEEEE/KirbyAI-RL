from gym.envs.registration import register
# The last part of the entry point is the environment class from Kirby/envs/kirby_env.py
register(
    id='kirby-v0',
    entry_point='Kirby.envs:KirbyEnv',
)
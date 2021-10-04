from gym.envs.registration import register

register(
    id='mapf-v0',
    entry_point='gym_mapf.envs:MapfEnv',
)

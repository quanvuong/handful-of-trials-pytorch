from gym.envs.registration import register

register(
    id='MBRLCartpole-v0',
    entry_point='env.cartpole:CartpoleEnv'
)

register(
    id='MBRLPusher-v0',
    entry_point='env.pusher:PusherEnv'
)

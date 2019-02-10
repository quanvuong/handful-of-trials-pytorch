from gym.envs.registration import register

register(
    id='MBRLCartpole-v0',
    entry_point='env.cartpole:CartpoleEnv'
)

register(
    id='MBRLPusher-v0',
    entry_point='env.pusher:PusherEnv'
)

register(
    id='MBRLReacher3D-v0',
    entry_point='env.reacher:Reacher3DEnv'
)

register(
    id='MBRLHalfCheetah-v0',
    entry_point='env.half_cheetah:HalfCheetahEnv'
)
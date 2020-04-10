from unityagents import UnityEnvironment

from continuous import logger
from continuous.agents import DDPG_Agent
from continuous.environment import UnityAdapter
import numpy as np

if __name__ == '__main__':
    logger.info('Reacher environment')
    env = UnityAdapter(
        UnityEnvironment(
            file_name="/Users/claudio/Dropbox/udacity-drl/code/drlnd-p2"
                      "-continuous-control/extras/Reacher_one.app.app",
            no_graphics=True,
        ),
        name="continuous_1")
    # state_size=33, action_size=4,
    agent = DDPG_Agent(env, seed=0,
                       max_train_episodes=2)
    scores = agent.train()
    env.close()
    logger.info(scores)

    # there are a LOT of meta parameters

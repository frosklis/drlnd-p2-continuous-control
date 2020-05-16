from unityagents import UnityEnvironment

from continuous import logger
from continuous.agents import DDPG_Agent
from continuous.environment import UnityAdapter
from continuous.problem import moving_average, loggable, moving_average_cross

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
                       is_finished_function=[
                           loggable(moving_average_cross, 100, 200),
                           loggable(moving_average, 100, 30)],
                       is_solved=loggable(moving_average, 100, 13), )
    scores = agent.train(
        max_training_episodes=5)
    env.close()
    logger.info(scores)

    # there are a LOT of meta parameters

from unityagents import UnityEnvironment

from bananas import logger
from bananas.agents_old import DQN_Agent, Double_DQN_Agent


if __name__ == '__main__':
    logger.info('Banana environment')
    env = UnityEnvironment(
        file_name="/Users/claudio/Dropbox/udacity-drl/code/deep-reinforcement-learning/p1_navigation/Banana.app",
        no_graphics=True,

    )
    agent1 = Double_DQN_Agent(state_size=37, action_size=4, seed=32, memory_a=1, memory_rank_based=False)
    agent2 = DQN_Agent(state_size=37, action_size=4, seed=32, memory_a=1, memory_rank_based=True)
    scores = agent1.train(max_episodes=2, score_success=20, env=env)
    scores = agent2.train(max_episodes=2, score_success=20, env=env)

    env.close()
    logger.info(scores)

    # there are a LOT of meta parameters


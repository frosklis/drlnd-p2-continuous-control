"""Basic problem elements definition"""
import inspect
import pickle
import tempfile
from functools import wraps
from typing import Tuple

import mlflow
import numpy as np

from bananas import logger
from .environment import Environment


def loggable(f, *args, **kwds):
    def params():
        total_args = [None] + list(args)
        arguments = inspect.signature(f).bind(*total_args, **kwds).arguments
        arguments.popitem(last=False)
        return arguments

    @wraps(f)
    def wrapper(agent):
        return f(agent, *args, **kwds)

    wrapper.log_dict = params
    return wrapper


def decay(agent, decay_factor, min_value=0):
    agent.epsilon = max(agent.epsilon * decay_factor, min_value)
    return agent.epsilon


def moving_average(agent, samples: int, min_value: float):
    return np.mean(agent.episode_scores[-samples:]) > min_value


def moving_average_cross(agent, samples_1: int, samples_2: int):
    if agent.training_episodes < max(samples_2, samples_1):
        return False
    avg_1 = np.mean(agent.episode_scores[-samples_1:])
    avg_2 = np.mean(agent.episode_scores[-samples_2:])

    return avg_2 > avg_1


class Agent:
    """Definition of an agent tied to an environment. The agent can interact with the environment and learn"""

    def __init__(self,
                 environment: Environment,
                 epsilon_initial_value=1,
                 is_finished_function=loggable(moving_average_cross, 20, 100),
                 is_solved=loggable(lambda x: False),
                 epsilon_update=loggable(decay, 0.995, min_value=0.01),
                 max_train_episodes=None,
                 seed: int = 0,
                 description='Generic Agent'):
        self.environment = environment
        self.training_episodes = 0
        self.state_size = environment.state_size
        self.action_size = environment.action_size
        try:
            is_finished_function = list(is_finished_function)
        except TypeError:
            is_finished_function = [is_finished_function]
        self._is_trained_functions = is_finished_function
        self.is_solved = is_solved.__get__(self)
        self._epsilon_update = epsilon_update.__get__(self)
        self.episode_scores = []
        self.epsilon = epsilon_initial_value
        self.description = description
        self.max_training_episodes = max_train_episodes
        self.seed = seed

    def __str__(self):
        return self.description

    def play_episode(self, epsilon: float = 0, learn: bool = True, max_actions: int = None) -> Tuple[float, int, bool]:
        total_reward = 0
        num_actions = 0
        while not self.environment.done:
            # Choose actions
            state = self.environment.state
            action = self.choose_action(state, epsilon=epsilon)

            # Take action
            reward = self.environment.step(action)
            total_reward += reward

            # Learn from the action
            if learn:
                self.learn_from_step(
                    state, action, reward,
                    self.environment.state, self.environment.done)

            # Break if we've played for too long
            num_actions += 1
            if max_actions is not None:
                if num_actions >= max_actions:
                    logger.warning(
                        'Episode interrupted because number of actions reached %d', max_actions)
                    break
        return total_reward, num_actions, self.environment.done

    def train(self):
        logger.info('Starting training for %s', str(self))
        mlflow.set_experiment(self.environment.name)

        with mlflow.start_run(run_name=str(self)) as active_run:
            run_id = active_run.info.run_id
            self.log_initial_setup()
            solved = False
            while not self.is_trained:
                self.environment.reset(train_mode=True)
                total_reward, num_actions, done = self.play_episode(learn=True, epsilon=self.epsilon)
                self.training_episodes += 1
                self.episode_scores.append(total_reward)
                mlflow.log_metrics({
                    'score': total_reward,
                    'num_actions': num_actions,
                    'done': 1 if done else 0,
                    'epsilon': self.epsilon,
                }, step=self.training_episodes)
                if not solved:
                    if self.is_solved():
                        solved = True
                        logger.info('Solved in %d episodes!!!', self.training_episodes)
                        mlflow.log_metric('episodes_to_solve', self.training_episodes)
                if self.max_training_episodes is not None:
                    if self.max_training_episodes <= self.training_episodes:
                        break
                if self.training_episodes % 100 == 0:
                    self.save(f'model_checkpoint_{self.training_episodes:04d}.pkl')
                self._epsilon_update()

            logger.info('%s trained finished after %d episodes',
                        str(self), self.training_episodes)

            mlflow.log_metrics({
                'training_episodes': self.training_episodes
            })
            mlflow.log_params({
                'is_trained': self.is_trained,
                'is_solved': solved,
            })
            # save final model
            self.save(f'model_final.pkl')
        return self.is_trained

    def choose_action(self, state, epsilon=0):
        return choose_action(self, state, epsilon=epsilon)

    def learn_from_step(self, state, action, reward, next_state, done):
        raise NotImplementedError

    @property
    def is_trained(self):
        if hasattr(self, '_is_trained'):
            return self._is_trained
        # Check functions one by one
        for f in self._is_trained_functions:
            if not f(self):
                return False
        return True

    def eval(self, state):
        raise NotImplementedError

    def log_initial_setup(self):
        params = dict()
        mlflow.log_params({
            'model class': self.model_class,
            'agent class': type(self)
        })
        params['agent'] = self.description
        params['max_training_episodes'] = self.max_training_episodes
        params['random_seed'] = self.seed
        for param, function in [('is_trained_function', self._is_trained_functions),
                                ('epsilon_update_function', self._epsilon_update),
                                ('is_solved_function', self.is_solved)]:
            try:
                functions = list(function)
            except TypeError:
                functions = [function]
            params[param] = ','.join([f.__name__ for f in functions])
            for f in functions:
                for k, v in f.log_dict().items():
                    params[f'{param} {f.__name__} {k}'] = v

        mlflow.log_params(params)

    def save(self, filename):
        try:
            logger.debug('Saving to %s', filename)

            self._save(filename)
            with open(filename, 'rb') as model_file:
                filebytes = model_file.read()
                pickled = {
                    'model class': self.model_class,
                    'agent class': type(self),
                    'file': filebytes
                }
            with open(filename, 'wb') as model_file:
                import pickle
                pickle.dump(pickled, model_file)

            mlflow.log_artifact(filename)

            import os
            os.remove(filename)
        except FileNotFoundError:
            logger.exception('Could not save agent.')

    def see_episode(self):
        self.environment.reset(train_mode=False)
        self.play_episode(epsilon=0, learn=False)

    @property
    def model_class(self):
        raise NotImplementedError

    def _save(self, filename):
        raise NotImplementedError

    @classmethod
    def load(cls, model_class, model_file, environment):
        raise NotImplementedError


def load(filepath, environment):
    logger.info('Loading model from %s', filepath)
    with open(filepath, 'rb') as file:
        pickled = pickle.load(file)

    agent_class_str = str(pickled['agent class'])
    model_class_str = str(pickled['model class'])
    logger.debug('agent class: %s', agent_class_str)
    logger.debug('model class: %s', model_class_str)

    def get_class(input):
        import importlib
        name = input.split("'")[1]
        parts = name.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    agent_class = get_class(agent_class_str)
    model_class = get_class(model_class_str)

    with tempfile.NamedTemporaryFile() as fp:
        fp.write(pickled['file'])
        agent = agent_class.load(model_class, fp.name, environment)

    return agent


def choose_action(agent: Agent, state, epsilon: float = 0):
    """Greedily chooses the best action based on epsilon

    with probability epsilon choose a random action, otherwise choose the best action

    Parameters
    ----------
    agent : Agent
        the agent that knows how to get action values from a state
    state : [type]
        [description]
    epsilon : float, optional
        epsilon value, in the interval [0, 1], by default 0 (choose best action)

    Returns
    -------
    int
        the chosen action
    """

    import random
    import numpy as np
    if random.random() > epsilon:
        # choose the best action
        action_values = agent.eval(state)
        action = np.argmax(action_values)
    else:
        # choose action randomly
        action = random.choice(np.arange(agent.action_size))
    return action

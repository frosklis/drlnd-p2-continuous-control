# Continuous control

## What is this

In this repository you can find several implementations of reinforcement
 learning algorithms that are suitable for agents that have a continuous
  action space, namely:
  
- deep Q-learning
- double deep Q-learning
- duelling networks

They have been tried for a custom Unity environment provided by [Udacity](https://www.udacity.com) based on the *Reacher* environment.

## Environment recreation

```bash
conda env create -f environment.yml
```

## Package installation

The following line will install the *bananas* package in your environment if executed from the root directory.

```bash
python setup.py install
```

Apart from pyTorch and python 3.6 (3.8 is not compatible), it will install other dependencies. In particular this code uses [mlflow](https://mlflow.org/) for experiment tracking. I wanted to try mlflow :)

## Usage

### As a python package.

```python
from unityagents import UnityEnvironment

from bananas.agents import Double_DQN_Agent


env = UnityEnvironment(
    file_name="Banana.app",    # Change to your location
)

agent = Double_DQN_Agent(state_size=37, action_size=4, seed=32, BUFFER_SIZE=int(
    1e5), BATCH_SIZE=64, GAMMA=0.99, TAU=1e-3, LR=5e-4, UPDATE_EVERY=4)
agent.train(max_episodes=2000, score_success=20, env=env)
env.close()
```

This mode of usage is showcased in the notebooks that train the model. 

### From the command line:

```bash
python -m bananas
```

### Caveats

- Although this code should work for any Unity Environment, it has only been tested for the Banana Environment.
- Not tested for GPU-enabled environments, some code may not work.
- It does not connect by default to any mlflow running server. If not configured, an mlruns folder will automatically be created.

## The bananas environment

The banana environment for which this code was created has the following characteristics.

It is an environment where there are blue bananas and yellow bananas (they fall from the sky apparently). You get 1 point for every yellow banana you touch and -1 point for every blue banana.

- episodic task (300 timesteps)
- the state is described as a vector of 37 float values
- there are four possible discrete actions to choose from
- the task is considered solved (as per problem statement) when the average reward for 100 consecutive runs is above 13.

However, even after the problem is solved the agent keeps training for a while. See the report for an explanation.  
 
## What is in this repo

This repository is structured according to the rubric for the first project (Navigation) in the [deep reinforcement learning nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

It has:
- This README.md file
- all modelling code is under bananas folder
- setup.py, requirements.txt ensure this is reproducible
- the trained model binary
- Navigation.ipynb: jupyter notebook showcasing the project
- a report
- mlruns folder contains the artifacts and run logs of several hyperparameter configurations

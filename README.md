# Continuous control

This is the last project from Udacity's deep learning nano degree where we have a robotic arm that we want to control. There is another version where there are 20 of such robotic arms


## Environment details

The environment is the Reacher environment, similar but not exactly equal to the one from Unity. The robotic arm has to "reach" for a target. To make things interesting, the target moves. The size and speed of the target are configurable. It is a double jointed arm.

For every timestep the robotic arm is in the target area, a reward of +0.1. There is no penalty for not being in the target area (just a reward of zero). Expressed in "human" terms, the goal of the robotic arm is to keep its hand in the target.

State and action space

- There are 33 continuous variables in the state space that express the cynematics of the system: position, rotation, linear and angular velocities, etc.
- The action space has 4 variables that represent the torque in each of the joints. All action values have to be between -1 and +1.

To consider the task solved, the agent must reach an average score of +30 over 100 consecutive episodes.

## To run this code

First create a python environment with python 3.6 and install the requirements from the requirements file. If using conda:

```bash
conda install -n myenv python=3.6
pip install -r requirements.txt
```

You also need jupyter notebook, install it if not already in your system. And make sure you add the environment you created to it.

```
conda install ipykernel
````

In order to install the environments, there is a makefile in the *extras* folder.

Enjoy!

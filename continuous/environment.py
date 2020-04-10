class Environment:
    @property
    def state(self):
        raise NotImplementedError

    @property
    def done(self):
        raise NotImplementedError

    def __init__(self, name: str):
        self.name = name

    def reset(self, **kwargs):
        raise NotImplementedError

    def step(self, action) -> float:
        raise NotImplementedError

    def __enter__(self):
        self.reset()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        raise NotImplementedError


class UnityAdapter(Environment):
    def __init__(self, unity_environment, name='unity environment'):
        super().__init__(name)
        self._environment = unity_environment
        # env_info = unity_environment.reset()[unity_environment.brain_names[0]]
        self.brain_name = unity_environment.brain_names[0]
        brain_name = self.brain_name
        self.state_size = unity_environment.brains[brain_name].vector_observation_space_size
        self.action_size = unity_environment.brains[brain_name].vector_action_space_size
        self.env_info = None

    def reset(self, **kwargs):
        self.env_info = self._environment.reset(**kwargs)[self.brain_name]

    def step(self, action) -> float:
        self.env_info = self._environment.step(action)[self.brain_name]

        return self.env_info.rewards[0]

    def close(self):
        self._environment.close()

    @property
    def done(self):
        return self.env_info.local_done[0]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._environment.close()

    @property
    def state(self):
        return self.env_info.vector_observations[0]


class Config:
    def __init__(self):
        self.device = 'cpu'
        self.seed = 0
        self.network = None
        self.optimizer = None
        self.memory = None
        self.noise = None
        self.hidden_units = None
        self.num_agents = 1
        
        self.actor_hidden_units = (64, 64)
        self.actor_network = None
        self.actor_optimizer = None
        self.actor_learning_rate = 0.004

        self.critic_hidden_units = (64, 64)
        self.critic_network = None
        self.critic_optimizer = None
        self.critic_learning_rate = 0.003
        
        self.tau = 1e-3
        self.weight_decay = 0
        self.states = None
        self.state_size = None
        self.action_size = None
        self.learning_rate = 0.001
        self.gate = None
        self.batch_size = 256
        self.buffer_size = int(1e5)
        self.discount = 0.999
        self.update_every = 16
        self.gradient_clip = None
        self.entropy_weight = 0.01
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995
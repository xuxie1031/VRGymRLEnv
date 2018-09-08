class Config:
    def __init__(self):
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.policy_fn = None
        self.discount = .99
        self.num_workers = 1

        self.target_network_mix = .001
        self.min_memory_size = 64

        self.use_gae = False
        self.gae_tau = .97
        self.entropy_weight = .01
        self.value_loss_weight = 1.0
        self.rollout_length = 5
        self.gradient_clip = .5

        self.target_network_update_freq = 10000
        self.exploration_steps = 50000
        self.double_q = False

        self.load_model = False
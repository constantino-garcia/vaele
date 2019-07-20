class Monitor(object):
    def __init__(self):
        self.epoch = -1
        self.batch = -1
        self.global_step = -1
        self.batch_data = None
        self.sde_lower_bound = None
        self.nnet_lower_bound = None
        self.stop_training = False
        self.session = None
        self.inferred_samples = None
        self.distributions = None
        self.training_state_means = None
        self.training_state_precs = None
        self.use_initial_state = None

    def update(self, epoch, batch, batch_data, sde_lower_bound, nnet_lower_bound, inferred_samples,
               distributions, training_state_means, training_state_precs, use_initial_state, rnn_initial_states):
        self.epoch = epoch
        self.batch = batch
        self.global_step += 1
        self.batch_data = batch_data
        self.sde_lower_bound = sde_lower_bound
        self.nnet_lower_bound = nnet_lower_bound
        self.inferred_samples = inferred_samples
        self.distributions = distributions
        self.training_state_means = training_state_means
        self.training_state_precs = training_state_precs
        self.use_initial_state = use_initial_state
        self.rnn_initial_states = rnn_initial_states
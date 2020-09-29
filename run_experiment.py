import tensorflow as tf
# Set up for debugging (uncomment if needed)
# tf.config.experimental_run_functions_eagerly(False)
# tf.config.experimental_functions_run_eagerly()
######################################### Experiment selection #########################################################
from experiments.settings.oscillator import experiment, svae_settings
########################################################################################################################
from utils.VaeFactory import GraphicalModelFactory
from utils.Trainer import Trainer
from utils.train import beep
from utils.schedulers import SigmoidScheduler, ConstantScheduler

vae = GraphicalModelFactory.build(**svae_settings, experiment=experiment)
trainer = Trainer(vae, experiment)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    1e-4, decay_steps=50, decay_rate=0.5, staircase=True
)
hoptimizers = tf.keras.optimizers.Adam(lr_schedule, clipvalue=10)
natgrad_lr_scheduler = ConstantScheduler(0.1)
kl_scheduler = SigmoidScheduler(1e-3, 1.0, 100, 200)

ckpt, ckpt_manager = trainer.configure_checkpoints()
monitor = trainer.configure_tensorboard_monitor(
    scalar_period=1, imgs_period=25, nb_images=2, do_phase_space=True
)
trainer.train(1000, hpars_optimizer=hoptimizers, lr_scheduler=natgrad_lr_scheduler,
              kl_scheduler=kl_scheduler, monitor=monitor, ckpt_manager=ckpt_manager,
              ckpt_period=25)
beep()

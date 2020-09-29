import matplotlib.pyplot as plt
import tensorflow as tf

from vaele_config import tf_floatx


def plot_scheduler(scheduler, max_epochs=10000):
    scheduled_values = []
    for epoch in range(max_epochs):
       scheduled_values.append(scheduler(epoch))
    plt.plot(scheduled_values)


class NatGradScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, maximum_learning_rate, growth_steps, name=None):
        super(NatGradScheduler, self).__init__()
        self.initial_learning_rate = tf.convert_to_tensor(initial_learning_rate, dtype=tf_floatx())
        self.maximum_learning_rate = tf.convert_to_tensor(maximum_learning_rate, dtype=tf_floatx())
        self.growth_steps = tf.convert_to_tensor(growth_steps, dtype=tf_floatx())
        self.name = name

        self._log_initial_learning_rate = tf.math.log(self.initial_learning_rate)
        self._log_maximum_learning_rate = tf.math.log(self.maximum_learning_rate)

    def __call__(self, step):
        with tf.name_scope(self.name or "NatGradSchedule") as name:
            p = step / self.growth_steps
            return tf.minimum(
                tf.exp(
                    self._log_initial_learning_rate +
                    (self._log_maximum_learning_rate - self._log_initial_learning_rate) * p
                ), self.maximum_learning_rate
            )

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'maximum_learning_rate': self.maximum_learning_rate,
            'growth_steps': self.growth_steps,
            'name': self.name
        }


class SigmoidScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, maximum_learning_rate, growth_steps,
                 midpoint, name=None):
        super(SigmoidScheduler, self).__init__()
        self.initial_learning_rate = tf.convert_to_tensor(initial_learning_rate, dtype=tf_floatx())
        self.maximum_learning_rate = tf.convert_to_tensor(maximum_learning_rate, dtype=tf_floatx())
        self.delta = self.maximum_learning_rate - self.initial_learning_rate
        self.growth_steps = tf.convert_to_tensor(growth_steps, dtype=tf_floatx())
        self.midpoint = tf.convert_to_tensor(midpoint, dtype=tf_floatx())
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "SigmoidScheduler"):
            return tf.sigmoid((step - self.midpoint) / self.growth_steps) * self.delta + self.initial_learning_rate

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'maximum_learning_rate': self.maximum_learning_rate,
            'delta': self.delta,
            'growth_steps': self.growth_steps,
            'midpoint': self.midpoint,
            'name': self.name
        }


class ConstantScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, name=None):
        super(ConstantScheduler, self).__init__()
        self.learning_rate = tf.constant(learning_rate, dtype=tf_floatx())
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "ConstantSchedule") as name:
           return self.learning_rate

    def get_config(self):
        return {
            'learning_rate': self.learning_rate,
        }



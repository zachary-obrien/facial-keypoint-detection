import tensorflow as tf
from tensorflow import greater, reduce_mean, reduce_sum
from tensorflow.math import log

# loss function that helps learning with heatmaps
# the issue with using MSE on the heatmaps is that most of the values are
# zero so once the model is outputting all zeros the gradients get very small.
# Adaptive wing loss weights the non-zero values of the ground truth heatmaps
# much higher.
# theory: https://arxiv.org/pdf/1904.07399v3.pdf
# code source: https://github.com/andrewhou1/Adaptive-Wing-Loss-for-Face-Alignment/blob/master/hourglasstensorflow/hourglass_tiny.py
def adaptive_wing_loss(labels, output):
    alpha = 2.1
    omega = 14
    epsilon = 1
    theta = 0.5
    with tf.name_scope('adaptive_wing_loss'):
        x = output - labels
        theta_over_epsilon_tensor = tf.fill(tf.shape(labels), theta/epsilon)
        A = omega*(1/(1+pow(theta_over_epsilon_tensor, alpha-labels)))*(alpha-labels)*pow(theta_over_epsilon_tensor, alpha-labels-1)*(1/epsilon)
        c1 = 1+pow(theta_over_epsilon_tensor, alpha-labels)
        c2 = omega*log(c1)
        c3 = theta*A
        C = c3-c2
        absolute_x = abs(x)
        losses = tf.where(greater(theta, absolute_x), omega*log(1+pow(absolute_x/epsilon, alpha-labels)), A*absolute_x-C)
        loss = reduce_mean(reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss

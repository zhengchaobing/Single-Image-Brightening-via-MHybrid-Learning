from ops import *
from config import *
import numpy as np
import tensorflow as tf

class net:

    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, train_phase):
        with tf.variable_scope(self.name):
            conv1 = prelu( conv("conv0", inputs, 64, 3, 1) )

            for d in np.arange(1, DEPTH - 1):
                Conv1 = prelu(batchnorm(conv("conv_" + str(d + 1), conv1, 64, 3, 1), train_phase, "bn" + str(d)))

            output1 = conv("conv" + str(DEPTH - 1), Conv1, IMG_C, 3, 1)

            output2 = tf.add(output1,inputs)

            Output  = tf.clip_by_value(output2,clip_value_min=0,clip_value_max=1)

            return Output
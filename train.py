import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.client import timeline

import model
import pipeline

tf.app.flags.DEFINE_integer('tile_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_integer('num_iter', 1, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('tune_from','',
                           """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('checkpoint_path', 'east_resnet_v1_50_rbox/', '')

FLAGS = tf.app.flags.FLAGS


def tower_loss(images, score_maps, geo_maps, training_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry = model.model(images, is_training=True)

    # calculate loss
    model_loss = model.loss(score_maps, f_score, geo_maps, f_geometry, training_masks)

    # add summary for tensorboard
    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)

    return model_loss


def get_train_op(loss):

    #variable_averages = tf.train.ExponentialMovingAverage(
      #  FLAGS.moving_average_decay, tf.train.get_global_step())
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())

    #with tf.control_dependencies(variables_averages_op):

    learning_rate = tf.train.exponential_decay(
        1e-4,
        tf.train.get_global_step(),
        2**16,
        0.9,
        staircase=False,
        name='learning_rate')

    optimizer = tf.train.AdamOptimizer(learning_rate)
    nn_vars = tf.get_collection( tf.GraphKeys.TRAINABLE_VARIABLES)
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=learning_rate, 
        optimizer=optimizer)
        #variables=nn_vars)
    #train_op = optimizer.minimize(loss, global_step = tf.train.get_global_step())

    return train_op


def input_fn():
    return pipeline.get_dataset(FLAGS.tile_size, FLAGS.batch_size)

def distribution_strategy(num_gpus=1):
    
    if num_gpus == 1:
        return tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
    elif num_gpus > 1:
        return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
    else:
        return none

def _get_init_pretrained( ):
    """Return lambda for reading pretrained initial model"""
    
    if not FLAGS.tune_from:
        return None
    
    # Extract the global variables
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES ) )
    
    ckpt_path=FLAGS.tune_from

    # Function to build the scaffold to initialize the training process
    init_fn = lambda sess: saver_reader.restore( sess, ckpt_path )

    return init_fn


def model_fn(features, labels, mode):

    tile = features['tile']
    geo = features['geo_map']
    score = features['score_map']
    
    train_mask = features['train_mask']

    scaffold = tf.train.Scaffold( init_fn=
                                  _get_init_pretrained( ) )

    model_loss = tower_loss(tile, 
                            geo, 
                            score,
                            train_mask)
    
    train_op = get_train_op(model_loss)

    return tf.estimator.EstimatorSpec( mode=mode, 
                                       loss=model_loss, 
                                       train_op=train_op,
                                       scaffold=scaffold)

def _get_config():
    config=tf.ConfigProto(allow_soft_placement=True)

    custom_config = tf.estimator.RunConfig(session_config=config,
                                           save_checkpoints_secs=30,
                                           train_distribute=distribution_strategy())

    return custom_config


def main(argv=None):
    

    classifier = tf.estimator.Estimator(config=_get_config(),
                                        model_fn=model_fn,
                                        model_dir=FLAGS.checkpoint_path)

    classifier.train(input_fn=input_fn)

   
if __name__ == '__main__':
    tf.app.run()

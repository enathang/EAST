import time
import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib import slim
import cv2

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 1, '')
tf.app.flags.DEFINE_integer('num_readers', 1, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 200, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'east_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')

import model
import icdar
import pipeline

FLAGS = tf.app.flags.FLAGS
gpus = list(range(len(FLAGS.gpu_list.split(','))))


def tower_loss(images, score_maps, geo_maps, training_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        f_score, f_geometry = model.model(images, is_training=True)

    # calculate loss
    model_loss = model.loss(score_maps, f_score, geo_maps, f_geometry, training_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary for tensorboard
    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

'''
def get_train_op(true_cls, pred_cls, true_geo, pred_geo, training_mask):
    loss = model.loss(true_cls, pred_cls, true_geo, pred_geo, training_mask)
    
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        tf.train.get_global_step(),
        FLAGS.decay_steps,
        FLAGS.decay_rate,
        staircase=FLAGS.decay_staircase,
        name='learning_rate')

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=FLAGS.momentum)

    train_op = tf.contrib.layers.optimize_loss(
        loss=loss
        global_step=tf.train.get_global_step(),
        learning_rate=learning_rate,
        optimizers=optimizers,
        variables=variables)

    return train_op
'''

def main(argv=None):
    # hyperparameters
    tile_size = 512
    batch_size = 1
    num_iter = 1000

    # data
    dataset = pipeline.get_batch(tile_size, batch_size)
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()
    print data

    # train
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for i in range(num_iter):
            #train_op = None
            sess.run(data)

        # test
        print sess.run(data)


def main1(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    # get training data
    data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                     input_size=FLAGS.input_size,
                                     batch_size=FLAGS.batch_size_per_gpu * len(gpus))
    data = next(data_generator)
    #input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    #input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
    input_images = tf.constant(np.asarray(data[0]))
    input_score_maps = tf.constant(np.asarray(data[2]))
    input_geo_maps = tf.constant(np.asarray(data[3]))
    input_training_masks = tf.constant(np.asarray(data[4]))

    #if FLAGS.geometry == 'RBOX':
    #    input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
    #else:
    #    input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 8], name='input_geo_maps')
    #input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    # establish gradient descent
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    # split the images among the gpus
    input_images_split = tf.split(input_images, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    # train model
    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        # for each gpu
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                # take in training data
                iis = input_images_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                
                #calculate loss
                total_loss, model_loss = tower_loss(iis, isms, igms, itms, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True
                
                # add gradient to update later
                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    # update gradients
    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    summary_op = tf.summary.merge_all()

    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # update batch norm
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    # load a pretrained model if it exists
    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
                                         input_size=FLAGS.input_size,
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))
        data = next(data_generator)

        # train the model for each step
        start = time.time()
        for step in range(FLAGS.max_steps):

            # get data to train
            #import cProfile, pstats, StringIO
            #pr = cProfile.Profile()
            #pr.enable()
            #data = next(data_generator)
            #pr.disable()
            #s = StringIO.StringIO()
            #ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            #ps.print_stats()
            #print s.getvalue()

            # do a forward pass
            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                             input_score_maps: data[2],
                                                                                             input_geo_maps: data[3],
                                                                                             input_training_masks: data[4]})
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            # print performance statistics
            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus))/(time.time() - start)
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                    step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             input_score_maps: data[2],
                                                                                             input_geo_maps: data[3],
                                                                                             input_training_masks: data[4]})
                summary_writer.add_summary(summary_str, global_step=step)
            # end


def testThroughput():
    data_generator = icdar.get_batch(num_workers=1,
                                     input_size=512,
                                     batch_size=1)
    data_generator
    
def testMaskSize():
    data_generator = icdar.get_batch(num_workers=1,
                                     input_size=512,
                                     batch_size=1)
    data = next(data_generator)
    input_image = data[0]
    input_score_maps = data[2]
    input_get_maps = data[3]
    input_training_masks = data[4]
    print input_score_maps
    encoded, size = cv2.imencode('.png', input_score_maps)
    print(size)

if __name__ == '__main__':
    tf.app.run()
    #testMaskSize()

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.client import timeline

import model
import pipeline

tf.app.flags.DEFINE_integer('tile_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size', 1, '')
tf.app.flags.DEFINE_integer('num_iter', 1, '')
FLAGS = tf.app.flags.FLAGS


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


def get_train_op(data):
    total_loss, model_loss = tower_loss(data['tiles'], 
                      data['geometry_maps'], 
                      data['score_maps'], 
                      data['training_masks'])
    optimizer = tf.train.AdamOptimizer(0.00001)
    train_op = optimizer.minimize(model_loss)

    return train_op


def getData(iterator):
    tile, geo, score, training_mask = iterator.get_next()
    data = {'tiles': tf.reshape(tile, [FLAGS.batch_size, FLAGS.tile_size, FLAGS.tile_size, 3]), 
            'geometry_maps':geo, 
            'score_maps': score, 
            'training_masks': tf.reshape(training_mask, [FLAGS.batch_size, FLAGS.tile_size/4, FLAGS.tile_size/4, 1])}

    return data


def main(argv=None):
    # data
    dataset = pipeline.get_dataset(FLAGS.tile_size, FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    data = getData(iterator)

    train_op = get_train_op(data)

    # train
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        
        #import cProfile, pstats, StringIO
        #pr = cProfile.Profile()
        #pr.enable()
        for i in range(FLAGS.num_iter):
            tl, ml, _ = sess.run([total_loss, model_loss, train_op], 
                                 options=run_options, 
                                 run_metadata=run_metadata)
            print 'tl:',tl, 'ml:', ml

            # profiling
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json','w') as f:
                f.write(ctf)
        #pr.disable()
        #s = StringIO.StringIO()
        #ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        #ps.print_stats()
        #print s.getvalue()


if __name__ == '__main__':
    tf.app.run()

import cv2
import importlib

# profiling
import cProfile
import pstats
import StringIO

import numpy as np
import tensorflow as tf
import random

from shapely.geometry.polygon import Polygon
from data_util import GeneratorEnqueuer

tf.app.flags.DEFINE_string('image_dir', 'data/maps/train/', '')
tf.app.flags.DEFINE_string('gt_dir', 'data/ground_truths/train/', '')
FLAGS = tf.app.flags.FLAGS


# creates a generator
def generator(tile_size, batch_size, scale):
    # import modules
    pt = importlib.import_module('pipelineTiling')
    pm = importlib.import_module('pipelineMaps')
    icdar = importlib.import_module('icdar')

    # get image and gt file names
    image_files = pt.getFilesFromDir(FLAGS.image_dir)
    gt_files = pt.getFilesFromDir(FLAGS.gt_dir)
    assert len(image_files) == len(gt_files)

    # load into memory
    images = [cv2.imread(_) for _ in image_files]
    groundtruths = [pt.parseGroundTruths(_) for _ in gt_files]
    groundtruth_points = [gt[0] for gt in groundtruths]
    groundtruth_polys = [gt[1] for gt in groundtruths]

    while True:
        tiles = list()
        ground_truths = list()
        geo_maps = list()
        score_maps = list()
        train_masks = list()
        # try to get data
        try:
            # get random historical map
            random_index = random.randint(0, len(images)-1)
            image = images[random_index]
            gt_points = groundtruth_points[random_index]
            gt_polys = groundtruth_polys[random_index]
            
            #disp = importlib.import_module('displayBoxes')
            #disp.displayImageModified(image, gt_points)

            # crop random tile
            #pr = cProfile.Profile()
            #pr.enable()
            tile, mod_ground_truth = pt.getRandomTile(image, gt_points, gt_polys, tile_size, scale)
            #pr.disable()
            #s = StringIO.StringIO()
            #ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            #ps.print_stats()
            #print s.getvalue()
            # display tile and bboxes to debug
            disp = importlib.import_module('displayBoxes')
            disp.displayImageModified(tile, mod_ground_truth)
            #input()
            tiles.append(tile)
            ground_truths.append(mod_ground_truth)

            # generate maps for that tile
            #pr = cProfile.Profile()
            #pr.enable()
            geo_map, score_map = pm.generate_maps((tile_size, tile_size), mod_ground_truth)
            #pr.disable()
            #s = StringIO.StringIO()
            #ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
            #ps.print_stats()
            #print s.getvalue()
            # display geo map and score map to debug
            train_mask = 0
            geo_maps.append(geo_map)
            score_maps.append(score_map)
            train_masks.append(train_mask)
            # return if batch is filled
            if len(tiles) == batch_size:
                yield tiles, ground_truths, geo_maps, score_maps, train_masks

        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
            
#ds = ds.map(func=pyfunc, num_thread=2)

#def pyfunc(tiles, ground_truth):
#    return tiles, ground_truths#, geo_maps, score_maps, train_masks
# prefetch large enough buffer size

def a():
    return gen(512, 1, 1)

def get_batch(num_threads, tile_size, batch_size):
    gen = generator(tile_size, batch_size, 1)
    # what is output data type?
    ds = tf.data.Dataset.from_generator(a, tf.int64, tf.TensorShape([None]))
    value = ds.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        sess.run(value)

    return value


if __name__ == '__main__':
    batch = get_batch(num_threads=1, tile_size = 1024, batch_size = 1)

import cv2
import importlib

import numpy as np
import tensorflow as tf
import random

from shapely.geometry.polygon import Polygon
from data_util import GeneratorEnqueuer

tf.app.flags.DEFINE_string('image_dir', 'data/maps/train/', '')
tf.app.flags.DEFINE_string('gt_dir', 'data/ground_truths/train/', '')
FLAGS = tf.app.flags.FLAGS


def generateTiles(tile_size):
    """
    Creates a tile generator that takes a map from memory, crops
    a tile of size tile_size, and finds all ground truths inside that tile
    
    param tile_size: the length of the tile side
    returns: a generator
    """
    # import modules
    pt = importlib.import_module('pipelineTiling')

    # get image and gt file names
    image_files = pt.getFilesFromDir(FLAGS.image_dir)
    gt_files = pt.getFilesFromDir(FLAGS.gt_dir)
    assert len(image_files) == len(gt_files)

    # load into memory
    images = [cv2.imread(_) for _ in image_files]
    groundtruths = [pt.parseGroundTruths(_) for _ in gt_files]
    groundtruth_points = [gt[0] for gt in groundtruths]
    groundtruth_polys = [gt[1] for gt in groundtruths]
    groundtruth_labels = [gt[2] for gt in groundtruths]

    while True:
        try:
            # get random historical map
            random_index = random.randint(0, len(images)-1)
            image = images[random_index]
            gt_points = groundtruth_points[random_index]
            gt_polys = groundtruth_polys[random_index]
            tile, mod_ground_truth = pt.getRandomTile(image, gt_points, gt_polys, tile_size)
            #disp = importlib.import_module('displayBoxes')
            #disp.displayImageModified(image, gt_points)

            yield tile, mod_ground_truth

        except Exception as e:
            import traceback
            traceback.print_exc()
            # continue


def generateMaps(tile, ground_truths):
    """
    Given a tile and ground truths, generate the geometry and score maps
    and the training mask

    param tile: the map tile
    param ground_truths: the list of ground truths that are contained in the tile
    returns: the tile and ground truths, plus the generated maps and mask
    """
    tile_size = FLAGS.tile_size
    pm = importlib.import_module('pipelineMaps')
    geo_map, score_map = pm.generate_maps((tile_size, tile_size), ground_truths)
    train_mask = np.ones((tile_size/4, tile_size/4, 1), dtype=np.float32)

    return tile, geo_map, score_map, train_mask


def generatorWrapper():
    """
    A wrapper for the tile generator

    returns: a generator
    """
    return generateTiles(FLAGS.tile_size)


def get_dataset(tile_size, batch_size):
    """
    Returns a batch generator

    param tile_size: the side length of the square tile
    param batch_size: the number of tiles per batch
    returns: a dataset object that can generate tiles
    """
    FLAGS.tile_size = tile_size
    prefetch_buffer_size = 1 # num of batches to prefetch
    ds = tf.data.Dataset.from_generator(generatorWrapper, 
                                        (tf.float32, tf.float32), 
                                        output_shapes=(tf.TensorShape([tile_size, tile_size, 3]), tf.TensorShape([4, 2, None])))
    ds = ds.apply(tf.contrib.data.map_and_batch(lambda tile, ground_truth: tf.py_func(generateMaps, 
                                                                                      [tile, ground_truth], 
                                                                                      [tf.float32, 
                                                                                       tf.float32, 
                                                                                       tf.float32, 
                                                                                       tf.float32]), 
                                                batch_size))

    return ds.prefetch(prefetch_buffer_size)


if __name__ == '__main__':
    tile_size = 512
    batch_size = 16
    batch = get_batch(tile_size = tile_size, batch_size = batch_size)
        

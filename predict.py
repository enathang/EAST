import cv2
import math
import os
import numpy as np
import tensorflow as tf

import lanms
import model
import data_tools

tf.app.flags.DEFINE_string('gpu_list', '0', 'List of available GPUs')
tf.app.flags.DEFINE_string('checkpoint_path', 'data/models/e3decay997cont/', 'Path to model checkpoint')
tf.app.flags.DEFINE_string('image_path', 'data/maps/test/D0042-1070002.tiff', 'Path to image or dir of images')
tf.app.flags.DEFINE_string('output_dir', 'output/test/', 'Desired location of prediction output')
tf.app.flags.DEFINE_bool('write_images', False, 'Whether to save images of predictions')
tf.app.flags.DEFINE_float('score_map_thresh', 0.5, 'Threshold for score map')
FLAGS = tf.app.flags.FLAGS


def save_boxes_to_file(boxes, image_name):
    """Saves the predicted boxes to a text file

    Parameters:
       boxes: a numpy array of box vertices
       image_name: the name of the corresponding image
    """
    res_file = os.path.join(
        FLAGS.output_dir,
        '{}.txt'.format(os.path.basename(image_name).split('.')[0]))

    print 'Saving',len(boxes),'boxes to',res_file
    with open(res_file, 'w') as f:
        for box in boxes:
            f.write('{},{},{},{},{},{},{},{},""\r\n'.format(
                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
            ))


def reconstruct_box(point, dist, theta):
    """
    Reconstructs a rotated rectangle from the corresponding point, distance to edges, and angle

    Parameters:
       point: the [y, x] coordinate of the pixel as np array
       dist: the distances from the pixel to [top, left, bottom, and right] edges as np array
       theta: the angle of rotation according to opencv
    Returns:
       box: a rotated rectangle defined by four vertices in bottom-left, bottom-right, 
            top-right, and top-left order
    """
    left_trans = np.asarray([-math.cos(theta), -math.sin(theta)])*dist[0]
    bottom_trans = np.asarray([math.sin(theta), -math.cos(theta)])*dist[1]
    right_trans = np.asarray([math.cos(theta), math.sin(theta)])*dist[2]
    top_trans = np.asarray([-math.sin(theta), math.cos(theta)])*dist[3]
    
    v0_trans = left_trans + bottom_trans
    v1_trans = bottom_trans + right_trans
    v2_trans = right_trans + top_trans
    v3_trans = top_trans + left_trans

    point = point*4 # compensates for downsampling in map construction
    v0 = point + v0_trans
    v1 = point + v1_trans
    v2 = point + v2_trans
    v3 = point + v3_trans
    box = np.asarray([v1, v2, v3, v0])
    
    return box


def convert_geometry_to_boxes(score_map, geo_map, score_map_thresh, nms_thresh=0.5):
    """Converts the predicted geometry map into rotated rectangles

    Parameters:
       score_map: the predicted score map
       geo_map: the predicted geometry map
       score_map_thresh: the minimum score to be counted as prediction
       nms_thres: the threshold for non-maximal supression
    Returns:
       boxes: a list of rotated rectangles
    """
    score_map = np.squeeze(score_map)
    geo_map = np.squeeze(geo_map)
    
    boxes = list()
    for i in range(len(score_map)):
        for j in range(len(score_map[0])):
            if (score_map[i, j] < score_map_thresh):
                continue
            point = np.asarray([i, j])
            dist = geo_map[i, j, 0:4]
            theta = -geo_map[i, j, 4] # negative turns from openCV to cartesian angle
            box = reconstruct_box(point, dist, theta)
            box = np.append(box, score_map[i, j])
            boxes.append(box)

    boxes = np.asarray(boxes)
    boxes = np.reshape(boxes, (len(boxes), 9))
    # only necessary for new_predict_test
    # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.5)
    
    return boxes


def create_tile_set(image, tile_shape):
    """Creates a set of tiles and shifts for those tiles

    Parameters:
       image: the image to tile
       tile_shape: a tuple of the shape of the tile

    Returns:
       tiles: the list of tiles
       shifts: the list of (y,x) shifts necessary to translate all points on the tile to the original image
    """
    tiles = list()
    shifts = list()
    tile_width = tile_shape[0]
    tile_height = tile_shape[1]
    im_width = len(image[0])
    im_height = len(image)

    # if the image is too small to tile, just return the image
    if (tile_height > im_height or tile_width > im_width):
        # cut off image to make sure it is able to be downsampled
        new_h = int(im_height/32)*32
        new_w = int(im_width/32)*32
        tile_0 = image[0:new_h, 0:new_w]
        tile_1 = image[0:new_h, im_width-new_w:im_width]
        tile_2 = image[im_height-new_h:im_height, 0:new_w]
        tile_3 = image[im_height-new_h:im_height, im_width-new_w:im_width]
        shift_0 = (0,0)
        shift_1 = (0, im_width-new_w)
        shift_2 = (im_height-new_h, 0)
        shift_3 = (im_height-new_h, im_width-new_w)
        return [tile_0, tile_1, tile_2, tile_3], [shift_0, shift_1, shift_2, shift_3]
    
    # calculate points
    h = int(math.floor(im_height/float(tile_height/2)))
    w = int(math.floor(im_width/float(tile_width/2)))
    list1 = range(h-1)
    list2 = range(w-1)
    list1 = [elem*tile_height/2 for elem in list1]
    list2 = [elem*tile_width/2 for elem in list2]
    if (im_height > tile_height):
        list1.append(im_height-tile_height)
    if (im_width > tile_width):
        list2.append(im_width-tile_width)

    for y in list1:
        for x in list2:
            tiles.append(image[y:y+tile_height, x:x+tile_width])
            shifts.append((y,x))

    if (len(tiles) == 0):
        tiles.append(image)
        shifts.append((0,0))

    return tiles, shifts


def convert_geometry_to_boxes(score_map, geo_map, score_map_thresh, nms_thresh=0.5):
    """Converts the predicted geometry map into rotated rectangles

    Parameters:
       score_map: the predicted score map
       geo_map: the predicted geometry map
       score_map_thresh: the minimum score to be counted as prediction
       nms_thres: the threshold for non-maximal supression
    Returns:
       boxes: a list of rotated rectangles
    """
    score_map = np.squeeze(score_map)
    geo_map = np.squeeze(geo_map)
    
    boxes = list()
    for i in range(len(score_map)):
        for j in range(len(score_map[0])):
            if (score_map[i, j] < score_map_thresh):
                continue
            point = np.asarray([i, j])
            dist = geo_map[i, j, 0:4]
            theta = -geo_map[i, j, 4] # negative turns from openCV to cartesian angle
            box = reconstruct_box(point, dist, theta)
            box = np.append(box, score_map[i, j])
            boxes.append(box)

    boxes = np.asarray(boxes)
    boxes = np.reshape(boxes, (len(boxes), 9))
    # only necessary for new_predict_test
    # boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.5)
    
    return boxes

def predict(sess, image_file, input_images, f_score, f_geometry, tile_shape):
    """Uses model to detect text in image

    Parameters:
       sess: the TensorFlow session
       image_file: the image to run through the model
       input_images: a TensorFlow placeholder
       f_score: a TensorFlow placeholder
       f_geometry: a TensorFlow placeholder
       tile_shape: a tuple of the tile shape
    """
    print image_file
    image = cv2.imread(image_file)[:, :, ::-1]
    image_tiles, shifts = create_tile_set(image, tile_shape)
    boxes = np.zeros((0,9))
    
    for i in range(len(image_tiles)):
        print 'predicting tile',i+1,'of',len(image_tiles)
        tile = image_tiles[i]
        shift = shifts[i]
        score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [tile]})
        tile_boxes = convert_geometry_to_boxes(score_map=score, geo_map=geometry, score_map_thresh = FLAGS.score_map_thresh, nms_thresh=0.5)
        if len(tile_boxes) != 0:
            #Shift boxes back into place
            shift = np.asarray([shift[0],shift[1],shift[0],shift[1],shift[0],shift[1],shift[0],shift[1],0])
            tile_boxes = tile_boxes[:,:]+shift
            boxes = np.concatenate((boxes, tile_boxes), axis=0)

    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.5)
    boxes = boxes[:, :8].reshape(-1, 4, 2)
    boxes = np.flip(boxes, axis=2) #IMPORTANT
    
    if boxes is not None:
        save_boxes_to_file(boxes, image_file)
        
    if FLAGS.write_images:
        import visualize_prediction
        visualize_prediction.save_image(image, boxes, image_file)
    
    
def main(argv=None):
    """Loads up a TensorFlow session, restores model, and detects text in image(s)"""
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        f_score, f_geometry = model.model(input_images, is_training=False)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver = tf.train.Saver()
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, model_path)
            
            image_file_list = data_tools.get_files_from_dir(FLAGS.image_path)
            for image_file in image_file_list:
                predict(sess, image_file, input_images, f_score, f_geometry, (4096, 4096))


if __name__ == '__main__':
    tf.app.run()

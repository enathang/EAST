import importlib
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import icdar
import eval





def mapsRegressionTest():
    # vars
    tile_size = 512

    # import modules
    icdar = importlib.import_module('icdar')
    pt = importlib.import_module('pipelineTiling')
    pm = importlib.import_module('pipelineMaps')
    
    img = cv2.imread('data/maps/train/D0041-5370006.tiff')
    gt = pt.parseGroundTruths('data/ground_truths/train/D0041-5370006.txt')
    points = gt[0]
    polygons = gt[1]
    labels = gt[2]
    tile, polys = pt.getRandomTile(img, points, polygons, tile_size)
    points_t = np.transpose(np.asarray(polys), (2, 0, 1))

    # get maps from icdar
    #icdar_score, icdar_geo, icdar_train = icdar.generate_rbox((tile_size, tile_size), points_t, labels)
    #icdar_score = icdar_score[::4, ::4, np.newaxis].astype(np.float32)
    #icdar_geo = icdar_geo[::4, ::4, :].astype(np.float32)

    # get maps from pipeline
    pl_score, pl_geo = pm.generate_maps((tile_size, tile_size), polys)

    # np.set_printoptions(threshold=np.nan)
    print 'pl',pl_geo
    #print 'icdar',icdar_geo

    import time, eval
    timer = {'net': 0, 'restore': 0, 'nms': 0}
    #icdar_boxes, timer = eval.detect(icdar_score, icdar_geo, timer)
    pl_boxes, timer = eval.detect(pl_score, pl_geo, timer)
    import displayBoxes
    displayBoxes.displayImageRegressionTesting(tile, pl_boxes)
    displayBoxes.displayImageRegressionTesting(tile, icdar_boxes)

    # compare

    # code taken from https://stackoverflow.com/questions/32105954/how-can-i-write-a-binary-array-as-an-image-in-python

    tile = tile[::4, ::4, :]
    #plt.imsave('tile.png', tile)
    #plt.imsave('icdar_score.png', icdar_score.reshape(128, 128), cmap=cm.gray)
    #plt.imsave('pipeline_score.png', pl_score.reshape(128, 128), cmap=cm.gray)

    '''
    plt.imshow(icdar_geo[:,:,3])
    plt.colorbar()
    plt.figure()
    plt.imshow(pl_geo[:,:,3])
    plt.colorbar()
    plt.show()
    '''

    print 'MaxDiff', np.squeeze(np.amax(np.amax(np.abs(icdar_geo - pl_geo),axis=0),axis=0))


if __name__ == '__main__':
    # 71, 92
    random.seed(150)
    np.random.seed(150)
    mapsRegressionTest()

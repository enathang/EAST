import importlib
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import icdar
import eval
import displayBoxes

def mapsRegressionTest():
    tile_size = 512
    pt = importlib.import_module('pipelineTiling')
    pm = importlib.import_module('pipelineMaps')
    
    # GET TILE AND GT
    img = cv2.imread('data/maps/train/D0017-1592006.tiff')
    gt = pt.parseGroundTruths('data/ground_truths/train/D0017-1592006.txt')
    points = gt[0]
    polygons = gt[1]
    labels = gt[2]
    tile, polys = pt.getRandomTile(img, points, polygons, tile_size)
    points_t = np.transpose(np.asarray(polys), (2, 0, 1))
    #score_map, geo_map = pm.generate_maps((512, 512), polys)
    #print geo_map.shape
    #input()

    # ICDAR GEO
    icdar_score, icdar_geo, icdar_train = icdar.generate_rbox((tile_size, tile_size), points_t, labels)
    icdar_score = icdar_score[::4, ::4, np.newaxis].astype(np.float32)
    icdar_geo = icdar_geo[::4, ::4, :].astype(np.float32)
    icdar_geo = icdar_geo[np.newaxis, :, :]
    icdar_score = icdar_score[np.newaxis, :, :]
    print icdar_geo.shape

    # PIPELINE GEO
    pl_score, pl_geo = pm.generate_maps((tile_size, tile_size), polys)
    pl_geo = pl_geo[np.newaxis, :, :]
    pl_score = pl_score[np.newaxis, :, :, :]
    print 'pl_score', pl_score.shape

    # BOXES
    timer = {'net': 0, 'restore': 0, 'nms': 0}
    icdar_boxes, timer = eval.detect(icdar_score, icdar_geo, timer)
    pl_boxes, timer = eval.detect(pl_score, pl_geo, timer)
    #pl_boxes = pl_boxes[:, :8].reshape((-1, 4, 2))

    # DISPLAY
    displayBoxes.displayImage(tile, points_t)
    displayBoxes.displayImageRegressionTesting(tile, pl_boxes)
    displayBoxes.displayImageRegressionTesting(tile, icdar_boxes)

    print 'MaxDiff', np.squeeze(np.amax(np.amax(np.abs(icdar_geo - pl_geo),axis=0),axis=0))


if __name__ == '__main__':
    # 71, 92
    random.seed(142)
    np.random.seed(142)
    mapsRegressionTest()

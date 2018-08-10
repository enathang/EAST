import importlib
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import icdar
import eval
import displayBoxes

def mapsTest():
    tile_size = 512
    pt = importlib.import_module('pipelineTiling')
    pm = importlib.import_module('semanticPipelineMaps')
    
    # GET TILE AND GT
    img = cv2.imread('data/maps/test/D0006-0285025.tiff')
    [points,polygons,labels] = pt.parseGroundTruths(
        'data/ground_truths/test/D0006-0285025.txt')
    tile, polys = pt.getRandomTile(img, points, polygons, tile_size)
    points_t = np.transpose(np.asarray(polys), (2, 0, 1))

    # PIPELINE GEO
    pl_score, pl_geo = pm.generate_maps((tile_size, tile_size), polys)
    pl_geo = pl_geo[np.newaxis, :, :]
    pl_score = pl_score[np.newaxis, :, :, :]
    print 'pl_score', pl_score.shape

    # BOXES
    timer = {'net': 0, 'restore': 0, 'nms': 0}
    pl_boxes, timer = eval.detect(pl_score, pl_geo, timer)
    #pl_boxes = pl_boxes[:, :8].reshape((-1, 4, 2))

    # DISPLAY
    displayBoxes.displayImage(tile, points_t)
    displayBoxes.displayImageRegressionTesting(tile, pl_boxes)


if __name__ == '__main__':
    # 71, 92
    random.seed(142)
    np.random.seed(142)
    mapsTest()

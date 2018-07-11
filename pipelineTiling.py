import os
import random
import importlib

import numpy as np
from shapely.geometry.polygon import Polygon

# takes in the top left coord of the tile, the tile size,
# and a list of Polygon ground_truths and returns a list
# of Polygon ground_truths that intersect the tile
def cropGroundTruths(x, y, tile_size, gt_coords, gt_polys):
    tile = Polygon([(x, y), (x, y+tile_size), (x+tile_size, y+tile_size), (x+tile_size, y)])
    intersects_tile = [tile.intersects(gt) for gt in gt_polys]
    intersects_tile = np.asarray(intersects_tile)
    # NOTE: Might introduce a bug when only one polygon intersection is found
    if (len(intersects_tile) == 1):
        print 'intersect_tile has len 1'
    keep_indices = np.squeeze(np.argwhere(intersects_tile))
    keep_gt = np.asarray(gt_coords[:,:,keep_indices])
    keep_gt[:,0,:] = keep_gt[:,0,:]-np.asarray([x])
    keep_gt[:,1,:] = keep_gt[:,1,:]-np.asarray([y])
    # TODO: Need to sort ground truths by area

    return keep_gt


# takes an image and crops a random tile, then gets
# the cropped ground truths
def getRandomTile(img, gt_coords, gt_polys, tile_size, scale):
    h = len(img)
    w = len(img[0])
    x = random.randint(0, w-tile_size)
    y = random.randint(0, h-tile_size)
    tile = img[y:y+tile_size,x:x+tile_size]
    ground_truth = cropGroundTruths(x, y, tile_size, gt_coords, gt_polys)

    return tile, ground_truth


# reads a ground truth file and returns a list of ground truth Polygons
def parseGroundTruths(file):
    icdar = importlib.import_module('icdar')
    points = list()
    polygons = list()
    file = open(file, 'r')
    for line in file.readlines():
        v = line[:-2].split(',')
        p1 = [float(v[0]), float(v[1])]
        p2 = [float(v[2]), float(v[3])]
        p3 = [float(v[4]), float(v[5])]
        p4 = [float(v[6]), float(v[7])]
        verts = [p1, p2, p3, p4]

        # icdar conversion and back
        # Assumes points are in clockwise order, which can
        # be changed in convert_data.py
        verts = np.asarray(verts)
        verts = icdar.sort_rectangle(verts)
        verts = verts[0]

        points.append(verts)
        poly = Polygon(verts)
        polygons.append(poly)

    print np.asarray(points).shape
    points = np.transpose(np.asarray(points), (1, 2, 0)) # Turn Nx4x2 to 4x2xN
    return [points, polygons]


# gets a list of the file names from the dir
def getFilesFromDir(dir):
    files = []
    if os.path.isdir(dir):
        names = os.listdir(dir)
        files = [dir + _ for _ in names]
    elif os.path.isfile(dir):
        files = [dir]
    else:
        assert False, 'invalid input directory or file'
    
    return sorted(files)

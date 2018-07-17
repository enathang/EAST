import os
import random
import importlib

import numpy as np
from shapely.geometry.polygon import Polygon


def cropGroundTruths(x, y, tile_size, gt_coords, gt_polys):
    """
    Takes a tile location+dimensions and finds all ground truths that
    intersect that tile

    param x: the x-coord of the top-left vertex of the tile
    param y: the y-coord of the top-left vertex of the tile
    param tile_size: the length of the tile side
    param gt_coords: a list of the ground truths in the form of the vertices
    param gt_polys: a list of the ground truths in the form of Shapely polygons
    returns: a list of shifted ground truths that intersect the tile
    """
    tile = Polygon([(x, y), (x, y+tile_size), (x+tile_size, y+tile_size), (x+tile_size, y)])
    intersects_tile = [tile.intersects(gt) for gt in gt_polys]
    intersects_tile = np.asarray(intersects_tile)
        
    keep_indices = np.squeeze(np.argwhere(intersects_tile))
    keep_gt = np.asarray(gt_coords[:,:,keep_indices])
    print len(keep_gt), len(keep_gt[0]), keep_gt
    keep_gt[:,0,:] = keep_gt[:,0,:]-np.asarray([x])
    keep_gt[:,1,:] = keep_gt[:,1,:]-np.asarray([y])
    # TODO: Need to sort ground truths by area

    return keep_gt


def getRandomTile(img, gt_coords, gt_polys, tile_size):
    """
    Takes an image, crops a tile, and then calculates thr ground truths

    param img: the image to take a tile from
    param gt_coords: the list of ground truths in the form of vertices
    param gt_polys: the list of ground truths in the form of Shapely polygons
    param tile_size: the length of the tile side
    returns: the tile and ground truths
    """
    h = len(img)
    w = len(img[0])
    x = random.randint(0, w-tile_size)
    y = random.randint(0, h-tile_size)
    tile = img[y:y+tile_size,x:x+tile_size]
    ground_truth = cropGroundTruths(x, y, tile_size, gt_coords, gt_polys)

    return tile, ground_truth


# reads a ground truth file and returns a list of ground truth Polygons
def parseGroundTruths(file):
    """
    Reads a ground truth file and returns a list, where the first element is
    the list of ground truths in the form of vertices and the second element
    is the list of ground truths in the form of Shapely objects

    param file: the name of the file to parse
    returns: points (4x2xN array) and polygons (list of Shapely polygons)
    """
    icdar = importlib.import_module('icdar')
    points = list()
    polygons = list()
    labels = list()
    file = open(file, 'r')
    for line in file.readlines():
        v = line[:-1].split(',')
        p1 = [float(v[0]), float(v[1])]
        p2 = [float(v[2]), float(v[3])]
        p3 = [float(v[4]), float(v[5])]
        p4 = [float(v[6]), float(v[7])]
        verts = [p1, p2, p3, p4]
        label = str(v[8])

        # Assumes points are in clockwise order, which can
        # be changed in convert_data.py
        verts = np.asarray(verts)
        verts = icdar.sort_rectangle(verts)
        verts = verts[0]

        points.append(verts)
        poly = Polygon(verts)
        polygons.append(poly)
        labels.append(label)

    print np.asarray(points).shape
    points = np.transpose(np.asarray(points), (1, 2, 0)) # Turn Nx4x2 to 4x2xN
    return [points, polygons, labels]


# gets a list of the file names from the dir
def getFilesFromDir(dir):
    """
    Gets a list of file names from a directory

    param dir: the name of the directory
    returns: a list of files in that directory
    """
    files = []
    if os.path.isdir(dir):
        names = os.listdir(dir)
        files = [dir + _ for _ in names]
    elif os.path.isfile(dir):
        files = [dir]
    else:
        assert False, 'invalid input directory or file'
    
    return sorted(files)

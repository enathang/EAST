import os
import numpy as np
import cv2
import math
from shapely.geometry.polygon import Polygon

def parse_boxes_from_json(file):
    """
    Reads a ground truth file and returns a list, where the first element is
    the list of ground truths in the form of vertices and the second element
    is the list of ground truths in the form of Shapely objects

    param file: the name of the file to parse
    returns: points (4x2xN array) and polygons (list of Shapely polygons)
    """
    import json
    points = list()
    polygons = list()
    labels = list()
    with open(file) as f:
        data = json.load(f)

    for entry in data:
        for item in entry['items']:
            labels.append(item['text'].encode('utf-8'))
            verts = item['points']

            # Assumes points are in clockwise order, which can
            # be changed in convert_data.py
            verts = np.asarray(verts, dtype=np.float32)
            verts = convert_polygon_to_rectangle(verts)

            points.append(verts)
            poly = Polygon(verts).convex_hull
            polygons.append(poly)

    points = np.transpose(np.asarray(points), (1, 2, 0)) # Turn Nx4x2 to 4x2xN
    return [points, polygons, labels]


def parse_boxes_from_text(file):
    """
    Reads a ground truth file and returns a list, where the first element is
    the list of ground truths in the form of vertices and the second element
    is the list of ground truths in the form of Shapely objects

    param file: the name of the file to parse
    returns: points (4x2xN array) and polygons (list of Shapely polygons)
    """
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
        verts = np.asarray(verts, dtype=np.float32)
        verts = convert_polygon_to_rectangle(verts)

        points.append(verts)
        poly = Polygon(verts)
        polygons.append(poly)
        labels.append(label)

    points = np.transpose(np.asarray(points), (1, 2, 0)) # Turn Nx4x2 to 4x2xN
    return [points, polygons, labels]


def get_files_from_dir(dir):
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


def distance(p1, p2):
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return dist


'''
Find the correct bottom left point of the box generated by opencv
and make it the first element of the list

points: the points of the box generated by opencv
target_point: the real bottom left point from json
'''
def set_correct_order(points, target_point):
    # find the left points of the long sides
    dis1 = distance(points[0], points[1])
    dis2 = distance(points[1], points[2])
    candidate_points = list()
    if (dis1 > dis2):
        candidate_points.append(0)
        candidate_points.append(2)
    else:
        candidate_points.append(1)
        candidate_points.append(3)

    # find closer candidate
    closest = 0

    dist1 = distance(points[candidate_points[0]], target_point)
    dist2 = distance(points[candidate_points[1]], target_point)
    if dist1 < dist2:
        closest = candidate_points[0]
    else:
        closest = candidate_points[1]

    # shift so closest point is first
    points = points[closest:]+points[:closest]
    return points

'''
poly: the list of points of the polygon read from json
show: whether to show debug info
'''
def convert_polygon_to_rectangle(poly):
    bottom_left = poly[0]
    box = cv2.minAreaRect(poly)
    points = cv2.boxPoints(box)
    
    # clockwise to ccw
    points = list(reversed(points.tolist()))
    points = set_correct_order(points, bottom_left)
    
    return points
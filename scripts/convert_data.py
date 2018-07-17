'''
Converts ground truths from arbitrary polygons to rotated rectangles
'''
import numpy as np
import cv2
import os
import importlib
import math
from shapely.geometry.polygon import Polygon

def loadGroundTruths(file):
    dict = np.load(file).item()
    polygons = list()
    for item in dict.keys():
        verts = dict[item]['vertices']
        verts.append(verts[0])
        polygon = np.array(verts, dtype=np.int32)
        polygons.append(polygon)

    return polygons


def loadGroundTruthFile(truth_file):
    ground_truth_list = []
    if os.path.isdir(truth_file):
        dir = sorted(os.listdir(truth_file))
        ground_truth_list = ['{}{}'.format(truth_file, i) for i in dir]
    elif os.path.isfile(truth_file):
        file = [truth_file]
        ground_truth_list = [_ for _ in file]
    else:
        assert False, 'invalid input'

    return ground_truth_list


def distance(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return math.sqrt(x*x + y*y)


# target point is the bottom left vertex of
# ground truth, which we always start with
def setCorrectOrder(points, target_point):
    # find closest point
    closest_point = 0
    shortest_dist = distance(points[0], target_point)
    for i in range(len(points)):
        dist = distance(points[i], target_point)
        if (dist < shortest_dist):
            closest_point = i
            closest_distance = dist

    # shift so closest point is first and points
    # are in counter clockwise order
    num_shifts = len(points)-closest_point-1
    print points[num_shifts:]
    print points[:num_shifts]
    points = points[num_shifts:]+points[:num_shifts]
    points = points.reverse()
    return points


def convertPolygonsToRectangles(polygons):
    rectangles = list()
    for poly in polygons:
        print poly.tolist()
        # don't convert if already rectangle
        if (len(poly) == 5):
            rectangles.append(poly.tolist()) # np.array
        else:
            rect = cv2.minAreaRect(poly)
            points = cv2.boxPoints(rect)
            points = setCorrectOrder(points.tolist(), poly[0])
            rectangles.append(points)
    
    return rectangles


def writeGroundTruths(name, polygons):
    name = "modified_" + name[0:-3] + "txt"
    print name
    file = open(name, "w+")
    for poly in polygons:
        output = ""
        for vert in poly:
            output += str(vert[0]) + ","
            output += str(vert[1]) + ","
        output += "###\n"
        file.write(output)
    file.close()


def main(truth_file):
    ground_truth_list = loadGroundTruthFile(truth_file)

    for file in range(len(ground_truth_list)):
      truths = loadGroundTruths(ground_truth_list[file])
      rect = convertPolygonsToRectangles(truths)
      writeGroundTruths(ground_truth_list[file], rect)


if __name__ == "__main__":
  main("ground_truths/")

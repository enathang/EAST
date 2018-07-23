import numpy as np
import cv2
import os
import importlib

import math
from shapely.geometry.polygon import Polygon

def loadGroundTruths(file):
    dict = np.load(file).item()
    polygons = list()
    labels = list()
    for item in dict.keys():
        verts = dict[item]['vertices']
        polygon = np.array(verts, dtype=np.int32)
        polygons.append(polygon)

        text = dict[item]['name']
        labels.append(text)

    return polygons, labels


def loadGroundTruthsFromJson(file):
    import json
    with open(file) as f:
        data = json.load(f)

    polygons = list()
    labels = list()
    for entry in data:
        for item in entry['items']:
            labels.append(str(item['text']))
            polygons.append(np.array(item['points'], dtype=np.int32))

    return polygons, labels


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


def rev(a):
    ret = list()
    length = len(a)
    for i in range(length):
        ret.append(a[length-i-1])

    return ret


def setCorrectOrder(points, target_point):
    # find closest point
    closest_point = 0
    shortest_dist = distance(points[0], target_point)
    for i in range(len(points)):
        dist = distance(points[i], target_point)
        if (dist < shortest_dist):
            closest_point = i
            closest_distance = dist

    # shift so closest point is first
    num_shifts = len(points)-closest_point-1
    points = points[num_shifts:]+points[:num_shifts]
    # switch from ccw to clockwise
    #points = rev(points)
    return points


def convertPolygonsToRectangles(polygons):
    rectangles = list()
    for poly in polygons:
        # don't convert if already rectangle
        if (len(poly) == 4):
            rectangles.append(rev(poly.tolist()))
        else:
            rect = cv2.minAreaRect(poly)
            points = cv2.boxPoints(rect)
            points = setCorrectOrder(points.tolist(), poly[0])
            rectangles.append(points)
    
    return rectangles


def writeGroundTruths(name, polygons, labels):
    name = "modified_" + name[0:-3] + "txt"
    file = open(name, "w+")
    for i in range(len(polygons)):
        poly = polygons[i]
        label = labels[i]
        output = ""
        for vert in poly:
            output += str(vert[0]) + ","
            output += str(vert[1]) + ","
        output += label+"\n"
        file.write(output)
    file.close()


def main(truth_file):
    ground_truth_list = loadGroundTruthFile(truth_file)

    for file in range(len(ground_truth_list)):
      truths, labels = loadGroundTruthsFromJson(ground_truth_list[file])
      rect = convertPolygonsToRectangles(truths)
      writeGroundTruths(ground_truth_list[file], rect, labels)


if __name__ == "__main__":
  main("ground_truths/")

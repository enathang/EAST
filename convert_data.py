import numpy as np
import cv2
import os
import importlib
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


def convertPolygonsToRectangles(polygons):
    rectangles = list()
    for poly in polygons:
        rect = cv2.minAreaRect(poly)
        points = cv2.boxPoints(rect)
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

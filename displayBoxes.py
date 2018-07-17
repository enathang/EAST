import numpy as np
import cv2
import os
import importlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry.polygon import Polygon


def loadBoxes(file):
    polygons = list()
    file = open(file, 'r')
    for line in file.readlines():
        v = line[:-2].split(',')
        p1 = [float(v[0]), float(v[1])]
        p2 = [float(v[2]), float(v[3])]
        p3 = [float(v[4]), float(v[5])]
        p4 = [float(v[6]), float(v[7])]
        verts = [p1, p2, p3, p4, p1]
        polygons.append(verts)

    return polygons


# code taken from https://stackoverflow.com/questions/34902477/drawing-circles-on-image-with-matplotlib-and-numpy
def displayImage(im, boxes):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(im)
    for n in range(len(boxes)):
        box = boxes[n]
        circ1 = Circle((box[0][0],box[0][1]), 4, color=(1,0,0))
        ax.add_patch(circ1)
        circ2 = Circle((box[1][0],box[1][1]), 4, color=(0,1,0))
        ax.add_patch(circ2)
        circ3 = Circle((box[2][0],box[2][1]), 4, color=(0,0,1))
        ax.add_patch(circ3)
        
    plt.show()


def displayImageModified(im, boxes):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(im)
    for n in range(len(boxes[0][0])):
        circ1 = Circle((boxes[0][0][n],boxes[0][1][n]), 4, color=(1,0,0))
        ax.add_patch(circ1)
        circ2 = Circle((boxes[1][0][n],boxes[1][1][n]), 4, color=(0,1,0))
        ax.add_patch(circ2)
        circ3 = Circle((boxes[2][0][n],boxes[2][1][n]), 4, color=(0,0,1))
        ax.add_patch(circ3)
        
    plt.show()


def main(img, truth):
    boxes = loadBoxes(truth)
    img = plt.imread(img)
    displayImage(img, boxes)


if __name__ == "__main__":
  main("maps/D0006-0285025.tiff", "modified_ground_truths/D0006-0285025.txt")

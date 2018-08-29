import numpy as np
import cv2
import os
import importlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from shapely.geometry.polygon import Polygon


def load_boxes(file):
    """Retrieves boxes from corresponding text file

    Parameters:
       file: the text file that contains the boxes
    Returns:
       polygons: a list of polygons
    """
    polygons = list()
    file = open(file, 'r')
    for line in file.readlines():
        v = line[:-2].split(',')
        p1 = [float(v[0]), float(v[1])]
        p2 = [float(v[2]), float(v[3])]
        p3 = [float(v[4]), float(v[5])]
        p4 = [float(v[6]), float(v[7])]
        verts = [p1, p2, p3, p4]
        polygons.append(verts)

    return polygons


def save_image(im, boxes, location):
    """Display first three vertices of each box on top of image
    
    Parameters:
       im: the image
       boxes: the boxes that corresponds to the image
    """
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(im)
    for n in range(len(boxes)):
        box = boxes[n]
        plt.plot([box[0][0], box[1][0]], [box[0][1], box[1][1]], linewidth=0.5, color='blue', scalex=False, scaley=False)
        plt.plot([box[1][0], box[2][0]], [box[1][1], box[2][1]], linewidth=0.5, color='red', scalex=False, scaley=False)
        plt.plot([box[2][0], box[3][0]], [box[2][1], box[3][1]], linewidth=0.5, color='red', scalex=False, scaley=False)
        plt.plot([box[3][0], box[0][0]], [box[3][1], box[0][1]], linewidth=0.5, color='red', scalex=False, scaley=False)

    save_as = location[0:-5]+'_predicted.png'
    print 'Saving image as', save_as
    plt.savefig(save_as)


# code taken from https://stackoverflow.com/questions/34902477/drawing-circles-on-image-with-matplotlib-and-numpy
def display_boxes(im, boxes):
    """Display first three vertices of each box on top of image
    
    Parameters:
       im: the image
       boxes: the boxes that corresponds to the image
    """
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(im)
    for n in range(len(boxes)):
        box = boxes[n]
        plt.plot([box[0][0], box[1][0]], [box[0][1], box[1][1]], linewidth=0.7, color='blue', scalex=False, scaley=False)
        plt.plot([box[1][0], box[2][0]], [box[1][1], box[2][1]], linewidth=0.5, color='red', scalex=False, scaley=False)
        plt.plot([box[2][0], box[3][0]], [box[2][1], box[3][1]], linewidth=0.5, color='red', scalex=False, scaley=False)
        plt.plot([box[3][0], box[0][0]], [box[3][1], box[0][1]], linewidth=0.5, color='red', scalex=False, scaley=False)
    
    plt.show()


def main(img, truth):
    boxes = load_boxes(truth)
    img = plt.imread(img)
    display_boxes(img, boxes)


if __name__ == "__main__":
  main("data/maps/validation/D0042-1070003.tiff", "output/test/D0042-1070003.txt")

import numpy as np
import os
import importlib
from shapely.geometry.polygon import Polygon

# Function taken from https://github.com/argman/EAST/issues/92
def polygon_iou(poly1, poly2):
  if not poly1.intersects(poly2):
    iou = 0
  else:
    try:
      inter_area = poly1.intersection(poly2).area
      union_area = poly1.area + poly2.area - inter_area
      iou = float(inter_area) / union_area
    except shapely.geos.TopologicalError:
      print('shapely.geos.TopologicalError occured, iou set to 0')
      iou = 0

  return iou


def loadGroundTruths(file):
    dict = np.load(file).item()
    polygons = list()
    for item in dict.keys():
        verts = dict[item]['vertices']
        verts.append(verts[0])
        polygon = Polygon(verts).convex_hull
        polygons.append(polygon)

    return polygons


def loadPredictions(file):
    polygons = list()
    file = open(file, 'r')
    for line in file.readlines():
        verts = line[:-2].split(',')
        p1 = tuple(verts[0:2])
        p2 = tuple(verts[2:4])
        p3 = tuple(verts[4:6])
        p4 = tuple(verts[6:8])
        verts = np.array([p1, p2, p3, p4, p1])
        polygon = Polygon(verts).convex_hull
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


def loadPredictionsFile(pred_file):
  predictions_list = []
  if os.path.isdir(pred_file):
    dir = sorted(os.listdir(pred_file))
    predictions_list = [pred_file+_ for _ in dir]
  elif os.path.isfile(pred_file):
    file = [pred_file]
    predictions_list = [_ for _ in file]
  else:
    assert False, 'invalid input'

  return predictions_list


def main(truth_file, pred_file):
    ground_truth_list = loadGroundTruthFile(truth_file)
    predictions_list = loadPredictionsFile(pred_file)
    assert len(ground_truth_list) == len(predictions_list)

    num_ground_truths = 0
    num_pred = 0
    num_correct = 0
    for k in range(len(predictions_list)):
      truths = loadGroundTruths(ground_truth_list[k])
      predictions = loadPredictions(predictions_list[k])
      iou_threshold = 0.5

      t, p, c = determineStats(truths, predictions, iou_threshold)
      num_ground_truths += t
      num_pred += p
      num_correct += c
      
    printStats(float(num_correct), float(num_pred), float(num_ground_truths), iou_threshold)


def determineStats(truths, predictions,  iou_threshold):
  num_correct = 0
  for t in range(len(truths)):
    for p in range(len(predictions)):
      if (polygon_iou(truths[t], predictions[p]) > iou_threshold):
        num_correct += 1

  return len(truths), len(predictions), num_correct


def printStats(num_correct, num_pred, num_ground_truths, threshold):
  precision = num_correct/float(num_pred)
  recall = num_correct/float(num_ground_truths)

  print '# predictions: ', num_pred
  print '# ground truths: ', num_ground_truths
  print '# correct: ', num_correct
  print 'iou threshold: ', threshold
  print 'precision: ', precision
  print 'recall: ', recall
  print 'fscore: ', 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
  main("ground_truths/", "output/")

import numpy as np
import os
from shapely.geometry.polygon import Polygon
import data_tools

# Function taken from https://github.com/argman/EAST/issues/92
def polygon_iou(poly1, poly2):
  """Calculates the IOU between two polygons

  Parameters:
     poly1: a polygon
     poly2: a polygon
  Returns:
     iou: the IOU of the two polygons
  """
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


def load_polygons_from_json(file):
  """Loads all the polygons from the corresponding JSON file
  
  Parameters:
     file: the JSON file
  Returns:
     data[1]: the list of polygons
  """
  import data_tools
  data = data_tools.parse_boxes_from_json(file)
  return data[1]


def load_polygons_from_text(file):
  """Loads all the polygons from the corresponding text file
  
  Parameters:
     file: the text file
  Returns:
     data[1]: the list of polygons
  """
  import data_tools
  data = data_tools.parse_boxes_from_text(file)
  return data[1]


def determine_stats(truths, predictions,  iou_threshold=0.5):
  """Calculates the number of correct predictions, number of predictions,
     and the number of ground truths

  Parameters:
     truths: the list of ground truth polygons
     predictions: the list of predicted polygons
     iou_threshold: the iou threshold for determining if correct prediction
  Returns:
     len(truths): the number of ground truths
     len(predictions): the number of predicted boxes
     num_correct: the number of correctly predicted boxes
  """
  num_correct = 0
  for t in range(len(truths)):
    pred_yet = False
    for p in range(len(predictions)):
      if (not pred_yet and polygon_iou(truths[t], predictions[p]) > iou_threshold):
        num_correct += 1
        pred_yet = True

  return len(truths), len(predictions), num_correct


def print_stats(num_correct, num_pred, num_ground_truths, threshold):
  """Calculated and prints the precision, recall, and f-score given
     basic statistics

  Parameters:
     num_correct: the number of correctly predicted boxes
     num_pred: the number of predicted boxes
     num_ground_truths: the number of ground truths
     threshold: the IOU threshold used to determine stats
  """
  precision = num_correct/float(num_pred)
  recall = num_correct/float(num_ground_truths)

  print '# predictions: ', num_pred
  print '# ground truths: ', num_ground_truths
  print '# correct: ', num_correct
  print 'iou threshold: ', threshold
  print 'precision: ', precision
  print 'recall: ', recall
  if (precision+recall != 0):
    print 'fscore: ', 2 * precision * recall / (precision + recall)
  else:
    print 'fscore: 0'
    

def main(truth_file, pred_file):
    """Loads up ground truth and prediction files, calculates and
       prints statistics

    Parameters:
       truth_file: the file or directory containing the ground truths
       pred_file: the file or directory containing the predictions
    """
    ground_truth_list = data_tools.get_files_from_dir(truth_file)
    predictions_list = data_tools.get_files_from_dir(pred_file)
    assert len(ground_truth_list) == len(predictions_list)

    num_ground_truths = 0
    num_pred = 0
    num_correct = 0
    for k in range(len(predictions_list)):
      truths = load_polygons_from_json(ground_truth_list[k])
      predictions = load_polygons_from_text(predictions_list[k])
      iou_threshold = 0.5

      t, p, c = determine_stats(truths, predictions, iou_threshold=iou_threshold)
      num_ground_truths += t
      num_pred += p
      num_correct += c
      print 'img', predictions_list[k]
      print 'prec', c/float(p)
      print 'recall', c/float(t)
      
    print_stats(float(num_correct), float(num_pred), float(num_ground_truths), iou_threshold)

    
if __name__ == "__main__":
  main("data/ground_truths/test/", "output/test/")

import pipeline
import predict
import data_tools
import cv2
import numpy as np

base_name = 'D0042-1070003'
map_file = 'data/maps/validation/' + base_name + '.tiff'
gt_file = 'data/ground_truths/validation/' + base_name + '.json'

gt = data_tools.parse_boxes_from_json(gt_file)
gt_points = gt[0]
print 'num gt', len(gt[1])
img = cv2.imread(map_file)
print 'generating maps'
tile, score_map, geo_map, train_mask = pipeline.generateMaps(img, gt_points, (7236, 5947))
score_map = np.squeeze(score_map)
print 'detecting boxes'
boxes = predict.detect(score_map, geo_map)

if boxes is not None:
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes = np.flip(boxes, axis=2) #IMPORTANT
        predict.save_boxes_to_file(boxes, base_name)

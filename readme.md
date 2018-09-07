# MapTD: Map Text Detector

### Introduction
MapTD is a text detection model for historical maps written in Tensorflow. The model is an extension of Argman's implementation of EAST []. 
Achievements:
 + 100x data input pipeline speed increase
 + Semantic-based model prediction (the model fully predicts the text rotation)
 + 74% recall on Rumsey collection of historical maps
 
### Install
TODO: Name dependencies and install instructions

### Train
Training is accomplished in train.py. Simply point the flags to the correct training data (data should be in JSON format with "points" and optional "text"label) and run the script. 

### Predict
To predict, set the flags correctly in predict.py and run. Predictions will be output in txt file, which can then be run through visualize_prediction.py to visualize. Alternatively, output txt file can be run through stats.py to calculate precision, recall, and f-score of the prediction.

### Trained model
TODO: Upload checkpoint of trained model

### Example outputs
TODO: Add images

### Acknowledgements
The author of the project gratefully acknowledges: Jerod Weinman, for mentorship; Abyaya Lamsal and Ben Gafford, for technical support; the Grinnell College Computer Science Department, for funding and support.

This work was supported in part by the National Science Foundation under grant Grant Number [1526350](http://www.nsf.gov/awardsearch/showAward.do?AwardNumber=1526350).


## OBJECT DETECTION

#### WHATS BEEN DONE
* working yolo5 and yolo8 classes
* working datamanager

#### WHATS BEING DONE
* trainer script
#### WHAT NEEDS TO BE DONE
* create functions for the models that output an array of bboxes for a given image
* get the models running on the pi's
* create an inference time testing function for every model
* fix mega detector class
	* loads the models
	* detects boxes
	* combines with classifier
	* figure out how to do evaluation
* create a global inference time tester
* decide what will be output by the processing button on the camera to determine what still needs to be done with the OD
## CLASSIFICATION

#### WHAT NEEDS TO BE DONE
* extract cut out images of meerkats (possibly in data manager)
* data loader function on data manger
* create boiler plate CNN class
* train the static classifier to detect meerkat and behaviours
* evaluate the model on the test images for behaviour and meerkat
* test the average inference time on pc and pi

## CAMERA
* define state machine
* create interfaces class that loads images to screen and 
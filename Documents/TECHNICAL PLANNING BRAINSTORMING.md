## CLASSIFICATION
GOAL: find all the meerkats in the image (possibly classify behaviour)
### DETECTING/LOCATING MEERKAT
1. TRAIN CLASSIFIER + LOCATOR
	* fine tune generically trained model architecture with meerkat class
	* train a classifier that can take in trap image, detect meerkat and classify as a meerkat
	* this will be very difficult because of lack of meerkat trap images, need a huge amount of data

2. USE GENERIC "OBJECT" DETECTING MODEL
	* basically don't try 
	* MEGADETECTOR -  trained by microsoft to classify animal/person/vehicle in camera trap data (well fitting for our use case)
	* IMAGENET - can use generically trained model architecture (YOLO/RESNET/CONVNET) to draw bounding boxes on potential "objects"
	* may not do great in kalahari scene
	* also no way of validating robustness

4. MOTION DETECTOR / TRACKING ALGORITHM
	* use a motion detection algorithm to find all pixels that contain moving objects
	* problem is that when meerkat stops it will no longer detect pixels
	* could use tracking algorithm that is robust to still meerkats
	* increases number of images that will have to be classified

## CAMERA TRAP

### HOUSING

### CAMERA + PROCESSOR + STORAGE

#### RASPBERRY PI 

<u>Advantages</u>
* More processing power for onboard processing
* Higher quality camera options
* Easier development environment
* More storage Capacity
* More options for night vision

<u>Disadvantages</u>
* Significantly more expensive
* Higher Power consumption
* Overkill amount of processing power if no onboard image processing
* Doesn't make sense to use a raspberry pi without doing onboard processing 
#### ESP32-CAM

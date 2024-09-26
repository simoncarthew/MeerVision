## INTRODUCTION
#### PROBLEM STATEMENT
#### OBJECTIVES
* build a robust camera trap 
* investigate the feasibility of computer vision techniques to extract visual information from the images\
#### SCOPE AND LIMITATIONS
#### PROJECT DEVELOPMENT

## LITERATURE REVIEW
#### KALAHARI MEERKATS
#### WILD LIFE MONITORING
#### IMAGE CLASSIFICATION
#### OBJECT DETECTION
#### BEHAVIOUR AND TRACKING
#### DEPTH ESTIMATION
#### ON-BOARD PROCESSING

## THEORETICAL BACKGROUND
#### MICRO CONTROLLERS AND CAMERAS
#### IMAGE CLASSIFICATION
* Basic CNN
* Different Architectures
#### YOLO
#### MEGADETECTOR
#### EVALUATION METRICS

## SYSTEM REQUIREMENTS
Deployment Times
Usability functions
Robustness
Minimum inference times to make onboard processing viable
## CAMERA TRAP DESIGN
#### DESIGN CHOICES
Processing unit
Camera
Triggering
Power
Housing
#### SUBSYSTEM DESCRIPTIONS
#### SYSTEMS DIAGRAM
## INVESTIGATED PROCESSING PIPELINES

### DATA
Acquisition
Type of images present in datasets
Benefits and drawbacks of each dataset
Processing (raw to desired format)
Diagrams of processing (directories and what they look like afterwards would look cool)
### DETECTION
#### YOLO
Potential benefits and drawbacks
Versions implemented and why
How they are implemented (libraries used)
What functionality is available
Training
#### MEGADETECTOR + CLASSIFIER
Brief intro on how the megadetector and classifier compliment each other to form meerkat specific detection
Potential benefits and drawbacks of this method
How the megadetector is implemented
How the classifier is implemented
### BEHAVIOUR CLASSIFICATION
How the previously explained implementation of the classifier can be extended to behaviour classification
Implement only static classifier as a means to find out if temporal information is required to classify Meerkat behaviour

## TESTING PROCEDURES
### CAMERA TRAP
All tests will be run using a script that can be run through the gui and the results are saved to a CSV upon completion.

| Tested Metric   | Varied parameters          | Procedure                                                                             |
| --------------- | -------------------------- | ------------------------------------------------------------------------------------- |
| Deployment Time | FPS (0.5 / 1 / 2)<br>Solar | Place in the garden with set parameters and start recording.                          |
| Inference Times | Models used<br>Device Used | Perform inference on all test images and get average inference times for every model. |


### YOLO DETECTION
#### DESCRIPTION
Varying sized yolo5 and yolo8 models are trained with different training parameters and training data.
#### HYPER PARAMETER TUNING
Training parameters and training data will be varied across the smallest versions of yolo5 and yolo8
Report on best parameters found for yolo5 and 8

|        | LR         | BATCH   | FREEZE | AUGMENT | PRETRAINED | PERCVAL | EPOCHS | MD_Z1_TRAINVAL | MD_Z2_TRAINVAL | IMG_SZ     | obs_no |
| ------ | ---------- | ------- | ------ | ------- | ---------- | ------- | ------ | -------------- | -------------- | ---------- | ------ |
| STD    | 0.01       | 32      | 0      | True    | True       | 0.2     | 50     | 1000           | 1000           | 640        | -1     |
| TUNING | 0.5 / 0.02 | 16 / 64 | 5 / 10 | -       | -          | -       | -      | 500 / 2000     | 500 / 2000     | 320 / 1280 | 0      |

#### MODEL SIZES
Best training parameters used to train larger models of yolo5 and yolo8
Plot mAP on val vs test to show overfitting if present (not necessary for all just show examples of overfit and not overfit)
Use evaluation metrics mAP50, mAP50-90, P, R from test images to interpret performance changes across models (Bar graphs probably make sense)

### MEGA TESTING
Best accuracy of different model architectures (ResNet/EfficientNet/etc.)
Testing the mAP, precision and recall of megaA and megaB on test set without classifier
Testing the mAP, precision and recall of MegaA and MegaB on test set with best classifier
### BEHAVIOUR CLASS
## RESULTS AND DISCUSSION
### CAMERA TRAP
DEPLOYMENT TIMES
![[IMG_0489.jpg]]
INFERENCE TIMES
![[IMG_0490.jpg]]
DISCUSSION
### YOLO DETECTION
HYPER PARAMETER TUNING
![[IMG_0491.jpg]]
![[IMG_0492.jpg]]
MODEL SIZE (with best tuned parameters)
![[IMG_0493.jpg]]
### MEGA TESTING
CLASSIFIER ACCURACY
![[IMG_0494.jpg]]

MEGA ACCURACY
![[IMG_0495.jpg]]
### BEHAVIOUR RECOGNITION
![[IMG_0496.jpg]]

### PIPELINE & MODEL CHOICES (BIG DISCUSSION)

## CONCLUSION

## FUTURE RECOMMENDATIONS

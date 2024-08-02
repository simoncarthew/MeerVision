## CLASSIFICATION
GOAL: find all the meerkats in the image (possibly classify behaviour)
### DETECTING/LOCATING MEERKAT
Use existing CNN architectures for object detection. Explore light weight models that may be able to run on-board and how they compare to larger models.
#### LARGE MODELS
- YOLOv5 (Trained on ImageNet)
- Megadetector (YOLOv5)
- ResNet
- ConvNeXt
- Vision Transformer
- SSD
- Refine Det

#### ONBOARD MODELS
* MobileNet-SSD
* TinyYOLO
* EfficientDet
* PP-YOLO-Tiny

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

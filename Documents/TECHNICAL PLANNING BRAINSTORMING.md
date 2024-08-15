## CLASSIFICATION
GOAL: find all the meerkats in the image (possibly classify behaviour)
### DETECTING/LOCATING MEERKAT
Use existing CNN architectures for object detection. Explore light weight models that may be able to run on-board and how they compare to larger models.
#### LARGE MODELS
- YOLOv5 (Trained on ImageNet)
- Megadetector (YOLOv5)
- ConvNeXt
- RetinaNet
- Vision Transformer (ViT)
- SSD
- Refine Det
- Detectron2
- DETA
#### ONBOARD MODELS
* MobileNet-SSD
* TinyYOLO
* EfficientDet
* PP-YOLO-Tiny
* SSD Lite
* RTMDet
- RT-DETR
- EdgeYOLO


## CAMERA TRAP

### HOUSING

### PROCESSING + CAMERA

#### PROCESSOR CHOICES

| DEVICE       | ADVANTAGES                                                                                                                                                                          | DISADVANTAGES                                                                                                                                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RASPBERRY PI | - More processing power for onboard processing<br>- Higher quality camera options<br>- Easier development environment<br>- More storage Capacity<br>- More options for night vision | - Significantly more expensive<br>- Higher Power consumption<br>- Overkill amount of processing power if no onboard image processing<br>- Doesn't make sense to use a raspberry pi without doing onboard processing |
| ESP32-CAM    | - cheap<br>- low power<br>- option for multiple angles because cheap                                                                                                                | - small storage capacity<br>- limited camera options<br>- minimal IR capabilities                                                                                                                                   |
#### CAMERA CHOICE

| DEVICE       | CAMERA     | ADVANTAGES                                        | DISADVANTAGES          |
| ------------ | ---------- | ------------------------------------------------- | ---------------------- |
| RASPBERRY PI | 5mp IR-Cut | - good image quality<br>- has automatic IR filter | - relatively expensive |
|              |            |                                                   |                        |

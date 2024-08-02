## MEERKATS

## WILDLIFE MONITORING

### Integrated animal monitoring system with animal detection and classification capabilities: a review on image modality, techniques, applications, and challenges
[LINK](https://link.springer.com/article/10.1007/s10462-023-10534-z)
#### TAKEAWAYS
#### QUOTES
<u>WHY WE NEED ANIMAL MONITORING</U>
* crucial for the well-being of both humans and animals
* poaching, fast habitat loss, and environmental degradation have resulted in enormous animal population declines and even extinction for many species of wild animals
* assessing population changes and modelling the effects of changes in human activities, and a variety of other stresses on wild animal populations
* animal populations over wide geographic areas is particularly difficult
<u>SYSTEMS</u>
* technical solutions for animal conservation -> low-power fast computation devices, parallel processing, and effective learning algorithms
* video monitoring system was mounted on the animals
* high-frequency radiotracking
* Global Positioning System (GPS) tracking
* satellite tracking with radio collars
* pyro-electric sensors
* wireless sensor networks
* little success, as the signals seem to be weaker in forest areas
* the camera trap technology emerged
* available for the researchers for data collection, besides being easy to use and maintain
* animal detection system plays a vital role include poaching, hunting, endangered species population count
<u>IMAGE ACQUISITION</u>
* camera traps, satellites, drones, and bio-loggers
* camera traps are predominantly used for data collection
* captures images upon sensing a physical phenomenon like motion or vibration. Passive Infrared (PIR) sensor is a commonly used motion-based sensor that senses infrared energy
* oldest of all image acquisition techniques is Biologgers
* problem is capturing the animal and fitting the sensor on the animal
* Drones, the unmanned aerial vehicle carries cameras on board to capture image and videos from locations
* on-board analytics in the drone that does the processing on the go and the results are alone sent to edge devices
* satellites are used in broader coverage and are usually used in the case of animal population estimation
<u>PHOTOGRAMMETRY TYPES</u>
* different levels like terrestrial, aerial, and satellite
* easier to process terrestrial images, based on the image resolution, the complexity of processing hyperspectral images increases
## OBJECT DETECTION AND CLASSIFICATION
### ANIMAL DETECTION PIPELINE FOR IDENTIFICATION
[LINK](https://ieeexplore.ieee.org/abstract/document/8354227?casa_token=F9bPKPMjy8kAAAAA:nykI_hwx2DbNLgdZ05gsd6rcuuqsXrnCIoCYB9sI4O2mjnjmHxb6RgdQiuVn1JbIbbIiBuTJp764)
##### QUOTES
* Computer vision-based methods are being used increasingly as tools to assist wild animal object recognition
* starting with the detection of animals in images and ending with identification decisions
* poor quality while others may show only parts of an animal due to self occlusion, or occlusion by other animals or vegetation

### DEEP ACTIVE LEARNING CAMERA TRAPS
[LINK](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13504)
##### TAKEAWAYS
* they argue that the problem with using just deep learning is that they are heavily reliant on labelled data and limited background in the images
* they propose using combination of object detection, transfer learning and active learning to overcome these issues
* use of convolutional neural network to extract info from camera trap images
* difference form normal neural network is that only a few outputs from previous layer are transferred to the next neurons input and weights are trained to learn a useful pattern
##### QUOTES

### Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning
[LINK](https://www.pnas.org/doi/full/10.1073/pnas.1719367115)

### Recognition of European mammals and birds in camera trap images using deep neural networks
[LINK](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1049/cvi2.12294)
## ON-BOARD COMPUTING

### OBJECT CLASSIFICATION AND VISUALISATION WITH EDGE AI FOR CUSTOMISED CAMERA TRAPS
[link]()
#### TAKEAWAYS
* give a good description of related work of edge-computing devices and transfer learning of small models
* process flow = train and validated on high end-computer (try different hyper parameters to get best) -> convert model to tensorflow lite
* they explored various transfer learning methods for training MobileNetV2
#### QUOTES
<u>GENERAL WILDLIFE MONITORING</U>
* possible to use the edge devices,such as Raspberry Pi as camera traps that can not only capture images and videos, but can also enable sophisticated image processing
* animal behaviour (Petso et al., 2021; Schindler and Steinhage, 2021), animal re-identification (Kuncheva et al., 2023) conservation (Dujon and Schofield, 2019), distribution (Ryo et al., 2021), and populationestimation
* automated processing of the image can take place off-site by transmitting the images (Chalmers et al., 2023; Nazir et al., 2017a; Whytock et al., 2023), retrieving the camera trap data (Wildlife Insights, n.d), or can take place on-boardthe device (Nazir et al., 2017b; Wei et al., 2020)
* avoid human-wildlife conflict or poaching
<u>MODEL OPTIONS</u>
* training of large DNN models can take days or weeks on state-of-the-art computers, whereas the inference only takes few seconds even on an edge device
* transfer learning, in which a pre-trained model trained on a large dataset, can be used in other classification problems
* large size pre-trained model, such as ResNet50 (98 MB), or comparatively smaller model, such as Mobile NetV2 (14 MB), useful for low power edge devices
<u>TRANSFER LEARNING</u>
* Transfer learning on the entire dataset was found to have limited value, however, it was concluded to be useful for projects with smaller dataset
* transfer learning approaches with small sized models, such as MobileNetV2,which had comparable performance to the large models
* improvements to the DNN architectures designed for mobile devices, with reduced model sizes, memory, and processing requirements, have made it possible to run state-of-the-art DNN models on mobile devices, or Raspberry Pi
* direct use of a pre-trained model means that a pre-trained model (results were poor)
* adding a classification layer or classifier(based on the classes in the new dataset)
* fine-tuning
	* train some or all of the layers of the pre-trained model, whereas the initial model layers could be used as such, thus reducing the overall time to train the mode
	* low-level features learnt by the initial layers, which are common to any classification problem are utilized
	* worked well if the dataset was small
	* Fine tuning requires a low learning rate
<u>MODELS</u>
* MobileNetV2
	* designed for low power devices
	* least size and low latency compared to other DNN models
	* don't have the best accuracy but size and latency make good choice for edge devices
	* smaller size of the model was achieved by depth separable convolutions with residual
* Model selection - selection of a model for an embedded application will be gov erned by the application latency needs, model size, and accuracy
* other models - EfficientNetV2B0, EfficientNetV2, Vision Transformers (ViT), MobileViT
* selected the Keras Tuner (O'Malley et al., 2019) whichprovides Random Search, Hyperband, and Bayesian Optimization algorithms for hyper parameter optimizing
<u>IMAGE VISUALISATION</u>
* difficult to explain why and how a particular prediction was made
* visualizations can help to answer the questions as to whichareas in the image contributed to the particular classification prediction
* explain the decision for the predicted class
* used tf-explain (tf-explain, n.d) to determine the visualizations with heat maps for the two techniques, Grad-CAM
* without re-training the models
TRAP
* used rpi zero
* infrared sensor
* power bank with solar panel (900000mAh)
* 5 mp camera

The literature review has led to a final [[PROPOSED SOLUTION]]
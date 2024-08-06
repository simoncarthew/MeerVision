## MEERKATS

## WILDLIFE BEHAVIOUR MONITORING TOOLS
#### Methods for wildlife monitoring in tropical forests: Comparing human observations, camera traps, and passive acoustic sensors
[LINK](https://conbio.onlinelibrary.wiley.com/doi/pdfdirect/10.1111/csp2.568)
* aim to give ecologists the information to make informed decision on monitoring techniques given the available resources for their particular project
* each method has its advantages and limitations
* most common is human, camera traps and acoustic sensors
* no universal selection -> {requires expert analysis of resources and desired outputs}
* proposes a selection pipeline of species coverage ? -> population metrics ? -> species identification ? -> resources required ?
* set objectives and maintain expectations regarding human and financial limitations
#### A Comprehensive Overview of Technologies for Species and Habitat Monitoring and Conservation
[LINK](https://academic.oup.com/bioscience/article-pdf/71/10/1038/40508174/biab073.pdf)
* technology played an important role on how we monitor the behaviour of animals in their natural habitats with radio tracking and camera traps
* common sensors used for animal monitoring (thermal sensors / atmospheric / optical / thermal / lidar / radar / acoustic / sonar / vibration / position and motion)
* devices used
	* cameras (remotely triggered or manually operated)
	* thermal imaging
	* passive acoustics
	* active sonar
	* terrestrial radar
	* terrestrial lidar
	* GPS trackers
* devices on vehicles
	* terrestrial vehicles
	* aquatic vehicles
	* airborne (drones / helicopters)
* animal-borne: biologging and biotelemetry = devices placed on or in animals for monitoring
* tracking = RF triangulation / GNSS / satellite based / inertial sensors / acoustic triangulation
* artificial intelligence
	* machine learning methods for data analysis
	* accelerating extraction of information from large data sources like camera trap images and sounds
	* no longer needs to be done manually
	* deep learning found success in identification tasks
	*  specialised skills often required to train models
	* TrailGuard AI camera trap for real-tie identification and alerts
	* advancements in hardware allowed for on-the-edge computing 
* low-cost computing like raspberry pi and Arduino has opened up world for DIY makers to experiment and help move conversation technology forward as they often don't require financial backing 
#### Integrated animal monitoring system with animal detection and classification capabilities: a review on image modality, techniques, applications, and challenges
[LINK](https://link.springer.com/article/10.1007/s10462-023-10534-z)
* poaching, habitat loss and environmental degradation has resulted in animal population decline
* one needs to regular monitor animals to asses the affects on animal populations
* low power fast computational devices, parallel processing and learning algorithms have helped improve this field
* monitoring involves a classification and detection process

#### Methods for Monitoring Large Terrestrial Animals in the Wild
[LINK]()
* 

#### Emerging technologies for behavioural research in changing environments
[LINK](https://pubmed.ncbi.nlm.nih.gov/36509561/)
* deep learning models have proven to be successful of classifying behaviour of animals using labelled images
* restricted to humans interpretations of animals behaviour by classifying their behaviour into discrete classes
* suggests the use of 3D pose estimation and unsupervised methods to help understand animal behaviour
* can be achieved with multi camera perspectives or single camera perspectives with labelled data
## COMPUTER VISION

### OBJECT DETECTION
#### A comprehensive review of object detection with deep learning
[LINK](https://www.sciencedirect.com/science/article/pii/S1051200422004298?casa_token=wo5kfs0DX4MAAAAA:hHYsJW_owpU0_c6q4XR6IA31C9Yls2TE__403V8FZ2UWhD98zEekdiyxhvRzgYnH9tt1HuyTL_YF)
* traditional CV methods
	* selection of region = entire image is scanned using multi-scale sliding window
	* extraction of features = hand crafted feature extracted used to extract essential components of desired object (not robust to all concievable light conditions etc.)
	* classifier (adaboost)
* describes all the good two-stage and regression models in great detail
* problems with OD
	* small object detection
	* multi-scale object detection
	* efficiency and scalability
	* generalisation
	* class imbalance
#### A systematic review of object detection from images using deep learning
[LINK](https://link.springer.com/content/pdf/10.1007/s11042-023-15981-y.pdf)
- used to extract information from images in the form of locating objects (surrounding in bounding box) and assigning them to specific class
- located by drawing bounding boxes
- autonomous vehicles, medical imaging, security surveillance, and robotics
- requires efficiency and robustness
- deep learning has proven to be very successful
- learns from a lots of data allowing it to be robust to different lighting, angles or occlusions of objects
- multiple objects locates and classifies multiple objects in one frame
- basic process is input image region proposals (multi-scale sliding process to scan the image for all possible regions that an object might be in), feature extraction (visual features extracted for semantic purposes), classification and output (co-ordinates and score of detected object)
- feature extraction and classification often performed by deep learning algorithms
- features are learnt by deep learning algorithms from set of input images
- pass input image through series of layers before reaching final classification layer
- CNN
	- no feature extraction
	- feed-forward
	- input -> convolutional layer -> pooling layer (reduce dimonesion) -> output layer
	- activation functions at neurons linearly transform input to output
	- stride filter type and padding
	- loss function, optimization functions and back propgation used to adjust filter values to adjust for calculated losses and subsequently train the network
- large datasets required for high quality DL models
- transfer learning uses previously trained model when only small datasets available
- two types = region proposal-based and classificatio/regression-based
- region proposal based = two stage, first stage scans whole image and then focus on region of interests, candidate regions are classified
	- RCNN
		* region proposal, CNN for freature extraction and classification/localization
		* generation of region proposals is hugely time consuming
	* SPPnet
		* removed restriced input size associated with RCNN
	* Fast RCNN
	* Faster RCNN
* regression-based OD = single stage to predict object instances
	* YOLO
		* frames it as single regression problem
		* image pizels to bounding boxes and box co-ordinates
		* split image into SxS grid
		* Each grid cell is responsible for predicting a certain number of bounding boxes and confidence scores for those boxes
		* YOLO predicts a fixed number of bounding boxes. Each bounding box prediction includes:
			- x and y coordinates for the center of the box, relative to the bounds of the grid cell.
			- Width and height of the box, relative to the whole image.
			- Confidence score for the box
			- Class probabilities for each class
	* SSD
		* typically uses a pre-trained image classification network (like VGG16) as the base network to extract feature maps from input images (backbone)
		* convolutional layers after the base network, which progressively decrease in size
		* multiple feature maps of different sizes where each of these maps is responsible for detecting objects of different sizes.
		* default (or anchor) boxes of different scales and aspect ratios at each feature map location
		* match various object shapes and size
		* each location on the feature maps, SSD predicts set of class scores (one for each class, including a background class) and set of offsets for the default boxes
* commonly used image datasets "PASCAL VOC, Microsoft COCO, ImageNet and OID (Open image Dataset)"
* ![[Pasted image 20240804195427.png|500]]
* commonly used frameworks are"Pytorch, Theano, Keras,and TensorFlow"
* common uses ""transportation object detection / pedestrian detection / face detection and recognition / remote sensing / text detection"
* most commonly used architectures are faster RCNN / YOLO and SSD
* feature extraction is the most time consuming
* best backbones "VGG / ResNet / MobileNet"
* challenges 
	* occlusions
	* small object detection
	* illumination
	* viewpint variation
	* deformation
	* speed and accurac tradeoff
	* training time
	* dataset available
* ML is promising approach to object detection

#### Transformer for object detection: Review and benchmark
[LINK](https://www.sciencedirect.com/science/article/pii/S0952197623012058?casa_token=XKQJhHnCLt8AAAAA:gMRM15ttMBklin-m81SnrLq6oCqnNdDkDsvfKboxQF3mO4ztO2p71DE2fPPSjsN-TiiFbFqrA5uV)
* transformers increasingly used in transformer based architechtures
* benchmarked models using COC2017
* issues with deep learning is balancing accuracy and efficiency, multi-scale objects, lightweight models
* after success in NLP transformers have been adapted to object detection (DETR and Swin Transformer)
* transformer highlights
	* amplifies the significant aspects of an image making it focus more on relevant features
	* parallel computations
* transformer limitations
	* require huge amounts of data for training
	* slower rate of convergence
	* inference is computationally expensive making it unfit for resource constrained settings
	* limited understanding of how it works
	* inefficient image-sequence information transformation
* CNN benefits
	* strong local feature extraction
	* lower computational complexity than transformer
* CNN limitations
	* rarely encodes relative feature positions
	* global feature extraction weaker
* transformers demonstrated superior performance over CNN
### USE IN CAMERA TRAPS
#### Assessing the potential of camera traps for estimating activity pattern compared to collar-mounted activity sensors
[LINK](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/wlb3.01263)
* goldern standard for animal activity tracking is a gps collar with accelerometer
* expensive and hard to maintain
* they compare the accuracy of network of camera traps compared to GPS collars to estimate the deil activity of lynx
* plotted animal population density in an area over time
* compared activity curves generated by camera traps and accelerometer data and concluded that with enough camera trap data one can estimate the activity of lynx with as much accuracy as the golden standard

#### DeepWILD
[LINK](https://www.sciencedirect.com/science/article/pii/S1574954123001243)
* extended the MegaDetector with the Faster RCNN Inceprion-ResNet-v2 to perform classification
* Counting by counting the number of animals in frame
* achieved 73.92% mAP for classification
* 96.88% mAP for detection
* better accuracy for high density classes

#### Recognition of European mammals and birds in camera trap images using deep neural networks
[LINK](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1049/cvi2.12294)
* created a system to detect and classify animals and mammals
* animal taxonomy, that is,genus,family ,order ,group,andclass names
* used megadetector for animal localization and tested ResNet, EfficientNetV2, Vision Transformer,Swin Transformer,and ConvNeXt for animal classification
* mAPs of 90.39% and 82.77% for best models
* convnext worked the best
* gives very good description of other works in detection and classification
* trained to recognise 25 mammal and 63 bird species that we consider in our studies
* use of augmentation to help with environment generalisation and avoid overfitting
* challenges
	* amount of data
	* data quality
	* adaptation to locations
	* species imbalance (some animals more abundant in dataset -> network ignore infrequently occurring species -> solved by re-entering the same image during training for smaller class sizes)

#### ANIMAL DETECTION PIPELINE FOR IDENTIFICATION
[LINK](https://ieeexplore.ieee.org/abstract/document/8354227?casa_token=F9bPKPMjy8kAAAAA:nykI_hwx2DbNLgdZ05gsd6rcuuqsXrnCIoCYB9sI4O2mjnjmHxb6RgdQiuVn1JbIbbIiBuTJp764)

#### DEEP ACTIVE LEARNING CAMERA TRAPS
[LINK](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13504)
* they argue that the problem with using just deep learning is that they are heavily reliant on labelled data and limited background in the images
* they propose using combination of object detection, transfer learning and active learning to overcome these issues
* use of convolutional neural network to extract info from camera trap images
* difference form normal neural network is that only a few outputs from previous layer are transferred to the next neurons input and weights are trained to learn a useful pattern
#### Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning
[LINK](https://www.pnas.org/doi/full/10.1073/pnas.1719367115)


## ON-BOARD COMPUTING
### TECHNIQUES AND MOST COMMON AND LIMITATIONS
#### Deep learning based object detection for resource constrained devices: Systematic review, future trends and challenges ahead
[LINK](https://www.sciencedirect.com/science/article/pii/S0925231223001388?casa_token=MNmtINU_F_IAAAAA:tQe_SNvKJi51ZOQ3nt961XiADaFjWixATm3mOu7CVyHew8yUQu_DFMayXaf0j6SWZgEVpfgU2uCr)
* investigate the current trends of techniques used to reduce the size of deep learning models for increased performance on resource constrained devices
* previously used two-stage detectors but later moved to single-stage detectors which don't rely on region proposals (YOLO/SSD)
* object classification models got better and used as backbones for object detectors
* need for real time applications moved towards trying to run them on resource constrained architectures
* are architectures that do this but precision isn't good enough for intolerable systems
* common uses of OD: remote sensing, aerial imagery, underwater imaging, medical imaging
* main battle is to balance limited battery high energy consumption, decreased computational abilities, limited memory all while maintaining high accuracy
* most commonly used framework is tensor flow lite which uses quantitation to reduce the size of models
* NVIDIA jetson and raspberry pi family very commonly used on the edge devices
* one generally requires large models fro good precision
* compression techniques are used to represent a model more efficiently with as little performance reductions as possible
	* pruning = removing redundant weights/neurons / blocks
	* quantisation = storing the models weights in lower bit representations (reduce the precision of the parameters)
* efficentdet has proven to work well on raspberry pi's
* knowledge distillation = larger model teaches smaller student to approximate its performance
* depth seperation / model scaling (efficientdet and mobilenet)
	* depth scaling = changing the number of layers
	* width scaling = adjusting number of neurons
	* resolution scaling = changing input image resolution
	* compound scaling = uniformly scales all dimensions using a set of predetermined scaling coefficients
* parameter reduction = reduce the number of parameters
* provides large <u>list of most commonly used</u> model architecture and whether they are suitable for mobile architectures
* must consider the balance between compression ratio and detection accuracy
### EXAMPLES OF ON-BOARD

#### OBJECT CLASSIFICATION AND VISUALISATION WITH EDGE AI FOR CUSTOMISED CAMERA TRAPS
[LINK]()
* give a good description of related work of edge-computing devices and transfer learning of small models
* process flow = train and validated on high end-computer (try different hyper parameters to get best) -> convert model to tensorflow lite
* they explored various transfer learning methods for training MobileNetV2

## OBJECT TRACKING


The literature review has led to a final [[PROPOSED SOLUTION]]
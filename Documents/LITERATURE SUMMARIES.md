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
lahoz-monfort_comprehensive_2021
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

## COMPUTER VISION

### GENERAL OBJECT DETECTION
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
### OBJECT DETECTION FOR CLASSIFICATION AND LOCALIZATION
#### Assessing the potential of camera traps for estimating activity pattern compared to collar-mounted activity sensors
[LINK](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/wlb3.01263)
* goldern standard for animal activity tracking is a gps collar with accelerometer
* expensive and hard to maintain
* they compare the accuracy of network of camera traps compared to GPS collars to estimate the deil activity of lynx
* plotted animal population density in an area over time
* compared activity curves generated by camera traps and accelerometer data and concluded that with enough camera trap data one can estimate the activity of lynx with as much accuracy as the golden standard
#### Detecting and monitoring rodents using camera traps and machine learning versus live trapping for occupancy modelling
[LINK]()
* compares physically trapping vs camera trapping for rodents
* difficulty associated with small non-charismatic species
* made use of transfer learning
* camera trap method proved to present a higher probability of detection and more accuracte representation of population
* made use of Yolov5x
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
#### Automatically identifying, counting, and describing wild animals in camera-trap images with deep learning
[LINK](https://www.pnas.org/doi/full/10.1073/pnas.1719367115)
* used object classification as means to extract information from camera trap images in the SS database
* treated counting animals as a classification problem (limited success)
* classified behaviour as classification with limited success
* tested models for detecting animals AlexNet (best 95.8% top-1 accuracy) NiN, VGG, GooLeNet, ResNet
* proved that its a good method for filtering images for animals or not but not more than that
#### DeepWILD
[LINK](https://www.sciencedirect.com/science/article/pii/S1574954123001243)
* extended the MegaDetector with the Faster RCNN Inceprion-ResNet-v2 to perform classification
* Counting by counting the number of animals in frame
* achieved 73.92% mAP for classification
* 96.88% mAP for detection
* better accuracy for high density classes
#### Deep Learning Object Detection Methods for Ecological Camera Trap Data
[LINK](https://ieeexplore.ieee.org/abstract/document/8575770)
* camera traps help ecologists monitor population species and animal behaviour in their natural environment without intruding on their natural activities
* camera traps respond to motion when animal enters the frame
* trained a model of YOLO v2 (92%) and Faster R-CNN (76.7%) old outdated models
* used Snapshot Serengeti (golden standard) qith 4,096 labeled images of 48 species classification

#### DEEP ACTIVE LEARNING CAMERA TRAPS
[LINK](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13504)
* use of convolutional neural network to extract info from camera trap images
* difference form normal neural network is that only a few outputs from previous layer are transferred to the next neurons input and weights are trained to learn a useful pattern
* they argue that the problem with using just deep learning is that they are heavily reliant on labelled data and limited background in the images
* they propose using combination of object detection, transfer learning and active learning to overcome these issues
* use transfer learning to minimise over fitting
* object detection is better than classification because they focus on areas of image where object in question is
	* less sensitive to image backgrounds
	* generalise more efficiently to new locations
	* but they are costly to get bounding boxes and expensive to run
	* they use transfer learning to overcome lack of BB
	* they used pre-trainde object detectors Faster RCNN and MegaDetector to crop images to animals
* transfer learning
	* leveraging the fact that early layers learn more general things
	* as you go deeper they are more specific to dataset
	* use pre-trained model and continue training it on new smaller dataset
	* use underlying supervised model and try to improve model by selecting most optimal training samples to be labeled
* active learning
	* minimise active work needed to train CV model
	* able to match state of art which used 3.2 million images with only 14100 images
	* iterates between training and asking trainer for some labels
* believe their system generalises better to new environment because it systematically removes background
* pipeline: OD to crop -> embedding layers to reduce dimensions -> active learning for classification

#### Accurate detection and identification of insects from camera trap images with deep learning
[LINK]()
* challenging task of identifying insect because they are small and prone to occlusions
* train and compare various YOLO models
* best performing was the YOLO v5
* average precision of 92.7% and a recall of 93.8% across 9 species of insect

#### Using motion-detection cameras to monitor foraging behaviour of individual butterflies
* 

### BEHAVIOUR MONITORING
#### Animal Behavior Analysis Methods Using Deep Learning: A Survey
* pose estimation
	* identifying and locating the position and orientation of objects
	* DeepLabCut and SLEAP
	* use pose estimation outputs to classify behaviour
	* used for behavioual profiling of robots
	* predicting locomotion behaviours
	* classify postures using nearest neighbour, random forest to stereotyped behaviours
	* used LSTM and 1D CNN to process trajectpries of poses for classification
* sensor based
* bioacoustics
* object detection
	* classify animal behaviours with object detectors that process single image (Faster RCNN / YOLO / MobileNetv2-SSD)
		* works well for positional behaviour
	* some people found that temporal motion important for some animal behavior classification (TempNet / SlowFast = spatio-temporal convolution )
	* tracking individuals using DeepSort
#### Computer Vision for Primate Behaviour Analysis in the Wild
[LINK]()
* suggests that monitoring animal behaviour using video footage involves  object detection, multi-individual tracking, action recognition and individual identification
* people have previously used deeplabcut but it has proven to not be robust to fluctuating conditions that come with monitoring animals in their natural environment
* their focus is on identification of monkeys and understand thier social networks and how they interact
* automatic behaviour recognition in natural environment is challenging task
* current keypoint technology for pose estimation makes assumptions and requires large amount of labelled data thus bounding boxes have proven to perform better in trying conditions (flexible and generalizable)
* animal detection
	* detect all animals of interest in an image
	* use ground truth bounding boxes
	* two types single stage (faster) and two stage (slower and more efficient)
* multi animal tracking
	* keep track of single animal over successive frames
	* normally seperated into two stages: detection and association
	* association stage links detected objects across frames
		* uses motion and visual characteristics
		* affinity cost measures the pairwise similarity between detected objects across frames
	* two types
		* track-by-detection: detection and association are separate steps (SORT)
		* track-by-query: detection and association are interlinked (MOTR)
* individual identification
	* closed set: set of known individuals (usually done with CNNs)
	* open set: identification of previously unseen individuals 
		* deep metric learning which rather learns features that distinguishes features form each other
* action understanding
	* spatio-temporal action detection classifies behaviour of anial at locaation and specific time
	* frame based approaches use temporal information for prediction
	* early backbones used 3D convolution to classify behavior
	* moving towards transformer networks
	* been shown that video processing techniques have proven to be better than trying to model motion with action related components

#### Improving wildlife tracking using 3D information
* suggests that tracking mechanisms that rely heavily on feature detection for tracking wont work well for similar looking animals like meekats
* 3D motion tracking requires stereo camera setup to sense extra dimension

#### Emerging technologies for behavioural research in changing environments
[LINK](https://pubmed.ncbi.nlm.nih.gov/36509561/)
* deep learning models have proven to be successful of classifying behaviour of animals using labelled images
* restricted to humans interpretations of animals behaviour by classifying their behaviour into discrete classes
* suggests the use of 3D pose estimation and unsupervised methods to help understand animal behaviour
* can be achieved with multi camera perspectives or single camera perspectives with labelled data
#### AI-Enabled Animal Behavior Analysis with High Usability: A Case Study on Open-Field Experiments
[LINK]()
* suggest that the problem with current behavioural analysis tool require a learning cost to implement
* researchers must use tools like python or SQL to interface with them
* bad human-machine intractability
* propose a platform that provides researchers with an easily navigate-able and flexible platform that allows them to tailor their behavioural analysis research to their specific scenario
* platform uses NLP to process database requests and allows for flexible choice of behavioural analysis algorithm
* provides pose estimation service and behaviour recognition/classification
* researchers can use and search for which algorithms they would like to use
* skeleton keypoint-based methods, optical flow information-based methods, depth image-based methods, and appearance contour-based methods
* use deeplabcut and yolox-pose for pose estimation
* deeplabcut
	* locate key points on limbs of experimental animals without labelling
	* uses Resnet as backbone
* yolox-pose
	* multi-person pose estimation
	* predicts bounding box and associated pose
	* different parameter scales for different processing architectures
* general two ways of behavioural recognition using key point co-ordinates
	* linear motion behaviours that can be matched by linear change co-ordinates of animals centre point
	* deep learning uses labelled key points to learn key-point change characteristics
* system successfully used on mouse 

#### Multi-animal pose estimation, identification and tracking with DeepLabCut
[LINK]()
* computer vision helpful for identifying counting and annotating animal behaviour
* problems are similar looking animals and occlusions
* pose-estimation is a crucial step in understanding fine-grained behaviour and deep learning is good at this
* to make robust one should annotate frames with closely interacting animals
* pose estimation: keypoint localization -> assembly -> tracking
* use unsupervised ID tracking
* data driven method for optimal skeleton assembly

#### An Automatic Behaviour Recognition System classifies animal behaviours using movements and their temporal context
[LINK](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6779137/pdf/nihms-1538868.pdf)
* complex and purposeful movements in flexible sequence
* need large amounts of data to identify and organise behaviour
* normally use behaviour recognition software
* can identify behaviour using configuration of body part or dynamics of movement
* proposed using spatiotemporal (combination of spatial and temporal) features to identify behaviour
* tested this on grooming flies
* suggest it would be useful for movement-based classification
* very controlled environment compared to camera trap images
* classifies movement over time which may be difficult for meerkats as some of their behaviour are stationary actions
* traditional CV methods rely on good spatial resolution, few occlusions, easy background subtraction or consistent appearance

#### Basic behaviour recognition of yaks based on improved SlowFast network
[LINK]()
* monitored basic behaviours of yaks (eating, lying, standing, walking)
* used new slowfast model for behaviour recognition
* backbone network used is ResNet50
* do this to help detect if a ya is sick (needed to be done in real time)
* people have used accelerameters to measure feeding patterns and other behavioural tendencies
* these methods are prone to damage and are stressful for animals
* computer vision techniques been proven to show high accuracy in detecting basic behaviours
* some methods do not take into account the spatio-temporal features of the video
* slow fast method uses two sampling paths that better account for the spatial features of a video over time
* does not require skeletal reconstruction
* video-based methods capture temporal information that allows for better classification of behaviour of livestock
* C3D and LSTM only classify whole video containing single animal
* their proposed method can classify multiple targets for each frame in a video
* dual path method helps reduce redundancy (slow path and fast path)
* proposed method has high accuracy
* best accuracy of 95.6%, recall of 91% and precision of 89.8% with the slowfast 3D resnet 50

#### Instance segmentation and tracking of animals in wildlife videos: SWIFT segmentation with filtering of tracklets
* track-by-detection has proven to perform better than track-by-query
* SWIFT
	* uses mask RCNN to detect masks of animals
	* these masks are then passed through SWIFT to keep track of unique animals
	* wont work well for meerkats because would require ground truth masks which is time consuming

#### Action Detection for Wildlife Monitoring with Camera Traps Based on Segmentation with Filtering of Tracklets (SWIFT) and Mask-Guided Action Recognition (MAROON)
* this is an extension of SWIFT that can be used for animal behaviour recognition
* problem with this is that it relies of swift which we have identified will not work cause no ground truth masks
* ectension of slowfast method for animal behaviour recognition

#### SUBTLE: An Unsupervised Platform with Temporal Link Embedding that Maps Animal Behaviour
* use unsupervised learning techniques to group skeletal reconstructions into clusters
* proved to provide more general groupings of animal behaviours that are not biased by human annotations
* used multiple camera system and object detection to estimate 3D structure
* wont be useful for monocular camera trap footage as it requires 3D skeltal reconstruction

#### Real-time sow behaviour detection based on deep learning
* used mobilenet as part of its proposed algorithm SBDA-DL to directly detect and classify three simple behaviours
* mobilenet is adapted from VGGNet and can process images 10 times faster
* uses ssd for object detection
* average precision (mAP) of a category is 93.4%, which can reach 7

#### Automated Behavior Recognition and Tracking of Group-Housed Pigs with an Improved DeepSORT Method
* used deepsort, YOLOX-S amd YOLOv5s for detection and tracking of pig behaviour
* used for heavily occluded, overlapping and illumination changes
* incorporate behaviour classification into tracking algorithm
* "YOLO v5s and YOLOX-S detectors achieved a high precision rate of 99.4% and 98.43%, a recall rate of 99% and 99.23"
* "multi-object tracking accuracy (MOTA), ID switches (IDs), and IDF1 of 98.6%,15, and 95.7%"
* "The MOT algorithms based on attention mechanisms are TransTrack [28] and TrackFormer" but they have higher inference times than SORT and deepsort
* tell the difference between standing and eating from aerial perspective

#### Animal Motion Tracking in Forest: Using Machine Vision Technology
* shown that video is an accurate way of monitoring animal movements and behaviour
* non-stressful, non-insvasive and cost-effective
* used YOLOv4 for animal detection
* SORT
	* rudimentary approach like Kalman filters and Hungarian algorithms
	* detection -> estimation (position of target in next frame) -> data association (cost matrix computed using iou between detection and estimation, solved using hungarian algorithm) -> creation and deletion of tracks (unique identities are created and destroyed based on IOUmin)
	* lots of id switches and failes in cases of occlusions
* used DeepSORT for animal tracking
	* tracks animals while assigning an ID to them
	* introduces deep learning into the standard SORT algorithm
	* uses both motion and appearance descriptors
	* the appearance descriptor reduces identity switches
	* more effective at tracking animals for longer periods and robust to id switches
	* discriminating feature embedding is trained offline on re-identification dataset using cosine metric learning
* results
	* deepsort has good speed
	* accuracy improved with fairMOT and centretrack
	* reduces ID switches
	* shown that yolo and deepsort is robust to poses and different views on animals with different backgrounds

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
import os
from CNNS import CNNS

# VARYING PARAMETERS
models = ["resnet50","mobilenet_v2","vgg16","efficientnet_b0","shufflenet_v2"]
batch_sizes = [2,4,8]
learning_rate = [0.05,0.01,0.005]
pretrained = [False,True]
img_size = [(64,64),(32,32)]

# STATIC PARAMETERS

# basic
raw_path = os.path.join("Data","Classification")
num_workers = 1
behaviour = False

# obs
obs_no = 400
obs_test_no = 50

# md
md_z1_trainval_no = 100
md_z2_trainval_no = 100
md_test_no = 0

#snap
snap_no=300
snap_test_no = 50

# RESULTS


# MAIN LOOP

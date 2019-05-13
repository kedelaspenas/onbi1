from torch.utils.data import Dataset, DataLoader
import os, cv2, numpy as np
import torch
from convcrf.convcrf import exp_and_normalize
from yolocrf import read_dataset, MAX_INSTANCES, train_files, test_files, YoloCRFDataset
from unetcrf import UNetCRFDataset
from keras.utils.np_utils import to_categorical

crf_conf = {
    'filter_size': 3,
    'blur': 8,
    'merge': True,
    'norm': 'none',
    'weight': 'vector',
    "unary_weight": 1,
    "weight_init": 0.2,

    'trainable': True,
    'convcomp': True,
    'logsoftmax': True,  # use logsoftmax for numerical stability
    'softmax': True,
    'final_softmax': False,

    'pos_feats': {
        'sdims': 3,
        'compat': 3,
    },
    'col_feats': {
        'sdims': 80,
        'schan': 13,   # schan depend on the input scale.
                       # use schan = 13 for images in [0, 255]
                       # for normalized images in [-0.5, 0.5] try schan = 0.1
        'compat': 10,
        'use_bias': True
    },
    "trainable_bias": True,

    "pyinn": False
}

class YOLOUNetCRFDataset(Dataset):
	def __init__(self, imagefiles, train=False, instance=False, max_instances=MAX_INSTANCES):
		self.imagefiles = imagefiles
		self.img_dims = 416
		self.train = train
		self.instance = instance
		self.max_instances = max_instances
		self.UNetCRFDataset = UNetCRFDataset(imagefiles, train, False, max_instances)
		self.YoloCRFDataset = YoloCRFDataset(imagefiles, train, instance, max_instances)

	def __len__(self):
		return len(self.imagefiles)

	def __getitem__(self, idx):

		file = self.imagefiles[idx]
		unetmasks, unetmaps = self.UNetCRFDataset._load_input(file)
		yolomasks, yolomaps = self.YoloCRFDataset._load_input(file)
		targets = self.UNetCRFDataset._load_target(file)

		return [torch.from_numpy(yolomasks*unetmasks).double(), torch.from_numpy(unetmasks).double(), torch.from_numpy(unetmaps).double()], torch.from_numpy(targets).double()

trainset = YOLOUNetCRFDataset(train_files, train=True, instance=True)
trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)

testset = YOLOUNetCRFDataset(test_files, train=False, instance=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

from torch.utils.data import Dataset, DataLoader
import os, cv2, numpy as np
import torch
from convcrf.convcrf import exp_and_normalize
from yolocrf import read_dataset, MAX_INSTANCES, train_files, test_files
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

class UNetCRFDataset(Dataset):
	def __init__(self, imagefiles, train=False, instance=False, max_instances=MAX_INSTANCES):
		self.imagefiles = imagefiles
		self.img_dims = 416
		self.train = train
		self.instance = instance
		self.max_instances = max_instances

	def __len__(self):
		return len(self.imagefiles)

	def __getitem__(self, idx):

		file = self.imagefiles[idx]
		masks, maps = self._load_input(file)
		targets = self._load_target(file)

		return [torch.from_numpy(masks).double(), torch.from_numpy(maps).double()], torch.from_numpy(targets).double()

	def _load_target(self, filename, instance=False):
		name = filename.split('/')[-1][:-4]
		target = cv2.imread(os.path.join('label', name + '.jpg'),0)
		if not instance:
			target[target > 0] = 1
			return cv2.resize(target, (self.img_dims, self.img_dims))
		else:
			img = cv2.threshold(cv2.resize(target, (self.img_dims, self.img_dims)), 127,255, cv2.THRESH_BINARY)
			ret, labels = cv2.connectedComponents(img[1])
			return labels

	def _load_input(self, filename):
		name = filename.split('/')[-1][:-4]
		data = np.load(os.path.join('intermediate2', 'train' if self.train else 'test', name + '.npz'))
		maps = cv2.imread(filename)
		maps = cv2.resize(maps, (self.img_dims, self.img_dims)).transpose(2,0,1)
		if not self.instance:
			# print (os.path.join('intermediate2', 'train' if self.train else 'test', name + '-semantic.jpg'))
			mask = cv2.resize(data['semantic'], (self.img_dims, self.img_dims), cv2.INTER_CUBIC)
			mask = np.expand_dims(mask, 0)
		else:
			mask = to_categorical(data['instance']).transpose(2,0,1)[1:]
			# in case greater than max_instances, get top in terms of area
			if mask.shape[0] > MAX_INSTANCES:
				zeros = np.count_nonzero(mask, axis=0)
				idx = np.argsort(zeros)[::-1]
				mask = mask[:50]
			# mask = np.pad(mask, ((0,self.max_instances-mask.shape[0]),(0,0),(0,0)), 'constant', constant_values=(0,))
		return [mask, maps]

trainset = UNetCRFDataset(train_files, True)
trainloader = DataLoader(trainset, batch_size=16, shuffle=False, num_workers=2)
trainset_instance = UNetCRFDataset(train_files, train=True, instance=True)
trainloader_instance = DataLoader(trainset_instance, batch_size=4, shuffle=True, num_workers=1)

testset = UNetCRFDataset(test_files, False)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
testset_instance = UNetCRFDataset(test_files, train=False, instance=True)
testloader_instance = DataLoader(testset_instance, batch_size=1, shuffle=False, num_workers=1)

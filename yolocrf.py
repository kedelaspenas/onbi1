from torch.utils.data import Dataset, DataLoader
import os, cv2, numpy as np
import torch
from convcrf.convcrf import exp_and_normalize

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

MAX_INSTANCES = 50

class YoloCRFDataset(Dataset):
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

	def _load_target(self, filename):
		name = filename.split('/')[-1][:-4]
		target = cv2.imread(os.path.join('label', name + '.jpg'),0)
		if not self.instance:
			target[target > 0] = 1
			return cv2.resize(target, (self.img_dims, self.img_dims))
		else:
			img = cv2.threshold(cv2.resize(target, (self.img_dims, self.img_dims)), 127,255, cv2.THRESH_BINARY)
			ret, labels = cv2.connectedComponents(img[1])
			return labels

	def _load_input(self, filename):
		name = filename.split('/')[-1][:-4]
		data = np.load(os.path.join('intermediate', 'train' if self.train else 'test', name + '.npz'))
		maps = np.squeeze(data['heatmaps'][:,:,:16]).transpose(2, 0, 1)
		if not self.instance:
			mask = cv2.imread(os.path.join('intermediate', 'train' if self.train else 'test', name + '-mask.png'), 0)
			mask = cv2.resize(mask, (416, 416))
			mask[mask > 0] = 1
			mask = np.array(mask, np.float)
			mask = np.expand_dims(mask, 0)
		else:
			mask = data['mask']
			mask_all = mask.sum(0) == 0
			# mask = np.pad(mask, ((1,self.max_instances-mask.shape[0]-1),(0,0),(0,0)), 'constant', constant_values=(0,))
			mask = np.pad(mask, ((1,0),(0,0),(0,0)), 'constant', constant_values=(0,))
			mask[0] = mask_all
		return [mask, maps]

def read_dataset(filename):
    f = open(filename)
    files = [i.strip() for i in f.readlines()]
    f.close()
    return files
    
train_files = read_dataset('train_neuroblastoma.txt')
trainset = YoloCRFDataset(train_files, True)
trainloader = DataLoader(trainset, batch_size=16, shuffle=False, num_workers=2)
trainset_instance = YoloCRFDataset(train_files, train=True, instance=True)
trainloader_instance = DataLoader(trainset_instance, batch_size=4, shuffle=True, num_workers=1)

test_files = read_dataset('test_neuroblastoma.txt')
testset = YoloCRFDataset(test_files, False)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
testset_instance = YoloCRFDataset(test_files, train=False, instance=True)
testloader_instance = DataLoader(testset_instance, batch_size=1, shuffle=False, num_workers=1)

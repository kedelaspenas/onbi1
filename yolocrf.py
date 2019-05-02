from torch.utils.data import Dataset, DataLoader
import os, cv2, numpy as np
import torch

crf_conf = {
    'filter_size': 11,
    'blur': 4,
    'merge': True,
    'norm': 'none',
    'weight': 'vector',
    "unary_weight": 5,
    "weight_init": 0.2,

    'trainable': True,
    'convcomp': False,
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

class YoloCRFDataset(Dataset):
	def __init__(self, imagefiles, train=False):
		self.imagefiles = imagefiles
		self.img_dims = 416
		self.train = train

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
		target[target > 0] = 1
		return cv2.resize(target, (self.img_dims, self.img_dims))

	def _load_input(self, filename):
		name = filename.split('/')[-1][:-4]
		data = np.load(os.path.join('intermediate', 'train' if self.train else 'test', name + '.npz'))
		maps = np.squeeze(data['heatmaps']).transpose(2, 0, 1)
		mask = cv2.imread(os.path.join('intermediate', 'train' if self.train else 'test', name + '-mask.png'), 0)
		mask = cv2.resize(mask, (416, 416))
		mask = np.array(mask, np.float)
		mask = np.expand_dims(mask, 0)
		mask[mask > 0] = 1
		return [mask, maps]

def read_dataset(filename):
    f = open(filename)
    files = [i.strip() for i in f.readlines()]
    f.close()
    return files
    
train_files = read_dataset('train_neuroblastoma.txt')
trainset = YoloCRFDataset(train_files, True)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)

test_files = read_dataset('test_neuroblastoma.txt')
testset = YoloCRFDataset(test_files, False)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
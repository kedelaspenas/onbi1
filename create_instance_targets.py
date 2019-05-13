import cv2, numpy as np
from yolocrf import train_files, YoloCRFDataset

dataset = YoloCRFDataset(train_files, train=True, instance=True)
for i in range(len(dataset)):
	input_, targets = dataset.__getitem__(i)
	print(targets.shape)
import torch
import os, cv2, numpy as np
import torch.optim as optim
import torch.nn as nn
from convcrf.convcrf import GaussCRF, exp_and_normalize
from yolocrf import trainloader, crf_conf

IMG_DIMS = (416,416)
NEPOCHS = 100000
crf = GaussCRF(crf_conf, IMG_DIMS, 1)
#crf = torch.load('crf-26.pt')
crf.to('cuda:0')

def custom_loss(output, target):
	mask = target == 1.0
	notmask = target != 1.0
	weight = 5
	loss = torch.sum((output[notmask] - target[notmask])**2) + torch.sum(weight*(output[mask] - target[mask])**2)
	return loss

# criterion = nn.CrossEntropyLoss()
pos_weight = torch.from_numpy(np.array([2.0], dtype=np.float32)).to('cuda:0')
criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
optimizer = optim.SGD(crf.parameters(), lr=1e-3, momentum=0.9)
#optimizer = optim.Adam(params = crf.parameters(), weight_decay=2.0)
def iou(o, target):
	for t in target:
		union = (o | t)
		intersection = (o & t)
		yield intersection.sum()/union().sum()

def match_output_to_target(output, target):
	#loop over each batch 
	batch = output.shape[0]
	for b in range(batch):
		output_one = output[b]
		# compute ious
		# loop over rois
		rois = target.shape[0]
		transpose_idx = np.zeros(target.shape[1])
		taken = np.zeros(target.shape[1])
		for r in range(rois):
			ious = np.array(i for i in iou(output_one[r], target[b]))
			idx = np.argsort(ious)
			ctr = 0
			while ctr < rois:
				if not taken[idx[ctr]]:
					transpose_idx[r] = idx[ctr]
					taken[idx[ctr]] = 1
					break
				ctr += 1
		output_one.transpose(transpose_idx)

for epoch in range(NEPOCHS):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs
		inputs, target = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		output = crf(inputs[0].to('cuda:0').float(), inputs[1].to('cuda:0').float())
		# target = target.unsqueeze(1)
		target = target.to('cuda:0').float()
		# match_output_to_target(output, target)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		# print statistics
		print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, loss))
		running_loss += loss
	print("Epoch: %d , Average Loss: %.8f" % (epoch + 1, running_loss/(i+1)))
	torch.save(crf, 'crf-'+str(epoch)+'.pt')	

PATH = 'crf.pt'
torch.save(crf, PATH)

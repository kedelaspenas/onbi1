import torch
import os, cv2, numpy as np
import torch.optim as optim
import torch.nn as nn
from convcrf.convcrf import GaussCRF as CRF, exp_and_normalize
from keras.utils.np_utils import to_categorical

BACKEND = 'yolounet'
if BACKEND == 'yolo':
	from yolocrf import crf_conf, trainloader_instance as trainloader, MAX_INSTANCES
elif BACKEND == 'yolounet':
	from yolounetcrf import crf_conf, trainloader, MAX_INSTANCES
	from instancecrf import InstanceCRF as CRF
else:
	from unetcrf import crf_conf, trainloader_instance as trainloader, MAX_INSTANCES

IMG_DIMS = (416,416)
NEPOCHS = 10000
# crf = GaussCRF(crf_conf, IMG_DIMS, 1)
crf = CRF(crf_conf, IMG_DIMS, MAX_INSTANCES) #max instances
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
criterion = nn.CrossEntropyLoss(reduction='mean')
# optimizer = optim.SGD(crf.parameters(), lr=1e-12, momentum=0.9)
for name, param in crf.named_parameters():
    if param.requires_grad:
        print (name)
optimizer = optim.Adam(params = crf.parameters(), lr=1e-5)
threshold = torch.tensor([0.5]).cuda()
def iou(o, target):
	target = torch.from_numpy(target).cuda().int()
	rois = o.shape[0]
	ious = np.zeros(rois)
	for t in range(rois):
		mask = target
		union = (o[t].float() > threshold).int() | mask
		intersection = (o[t].float() > threshold).int() & mask
		ious[t] = intersection.sum()/union.sum()
	return ious

def match_output_to_target(output, target):
	#loop over each batch 
	batch = output.shape[0]
	for b in range(batch):
		output_one = output[b]
		target_one = to_categorical(target[b].cpu()).transpose(2,0,1)
		# compute ious
		# loop over rois
		out_rois = output_one.shape[0]
		target_rois = target_one.shape[0]
		match_idx = np.zeros(target_rois, dtype='int')
		taken = np.zeros(out_rois)
		# loop over the targets
		for r in range(target_rois):
			ious = iou(output_one, target_one[r])
			idx = np.argsort(ious)[::-1]
			ctr = 0
			while ctr < out_rois:
				if not taken[idx[ctr]]:
					match_idx[r] = idx[ctr]
					taken[idx[ctr]] = 1
					break
				ctr += 1
		target_one = target_one[np.argsort(match_idx)]
		target[b] = torch.from_numpy(np.argmax(target_one, axis=0))
# 		target[b] = torch.from_numpy(np.zeros(target[b].shape, dtype='float32'))
# 		for r in range(target_rois):
# 			target[b][target_one[r]] = float(match_idx[r])
	return output, target

for epoch in range(NEPOCHS):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs
		inputs, target = data
		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		if BACKEND == 'yolounet':
			output = crf(inputs[0].to('cuda:0').float(), inputs[1].to('cuda:0').float(), inputs[2].to('cuda:0').float())
		else:
			output = crf(inputs[0].to('cuda:0').float(), inputs[1].to('cuda:0').float())
		# target = target.unsqueeze(1)
		target = target.to('cuda:0')
		# print(target.max(), output.shape)
		# if output is less than target, pad
		_, target = match_output_to_target(output, target.float())
		# loss = criterion(output[:,:target.max().int(),:,:], target.long())
		# print(target.max().int(),output.shape)
		if target.max().int()-output.shape[1] >= 0:
			output = nn.functional.pad(output, (0,0,0,0,0,target.max().int()-output.shape[1]+1,0,0), 'constant', 0)
		# print(target.max().int(),output.shape[1])
		loss = criterion(output, target.long())
		loss.backward()
		optimizer.step()

		# print statistics
		print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, loss))
		running_loss += loss
	print("Epoch: %d , Average Loss: %.8f" % (epoch + 1, running_loss/(i+1)))
	if epoch%20 == 0:
		torch.save(crf, BACKEND + 'crf-'+str(epoch)+'.pt')	

PATH = BACKEND + 'crf.pt'
torch.save(crf, PATH)

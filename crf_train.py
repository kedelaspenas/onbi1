import torch
import os, cv2, numpy as np
import torch.optim as optim
import torch.nn as nn
from convcrf.convcrf import GaussCRF
from yolocrf import trainloader, crf_conf

IMG_DIMS = (416,416)
NEPOCHS = 100
crf = GaussCRF(crf_conf, IMG_DIMS, 1)
crf.to('cuda:0')

def custom_loss(output, target):
	target = torch.unsqueeze(target, dim=1)
	mask = target == 1.0
	notmask = target != 1.0
	weight = 5
	loss = torch.mean((output[notmask] - target[notmask])**2) + torch.mean(weight*(output[mask] - target[mask])**2)
	return loss

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss(reduce='sum')
optimizer = optim.SGD(crf.parameters(), lr=1e-10, momentum=0.5)

for epoch in range(NEPOCHS):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs
		inputs, target = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = crf(inputs[0].to('cuda:0').float(), inputs[1].to('cuda:0').float())
		target = target.unsqueeze(1)
		loss = criterion(outputs, target.to('cuda:0').float())
		# loss = custom_loss(outputs, target.to('cuda:0').float())
		loss.backward()
		optimizer.step()

		# print statistics
		print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, loss))
		running_loss += loss
	print("Epoch: %d , Average Loss: %.8f" % (epoch + 1, running_loss/(i+1)))
	torch.save(crf, 'crf-'+str(epoch)+'.pt')	

PATH = 'crf.pt'
torch.save(crf, PATH)
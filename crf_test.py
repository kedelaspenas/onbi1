from convcrf.convcrf import GaussCRF
from torch.utils.data import Dataset, DataLoader
import os, cv2, numpy as np
import torch
from yolocrf import trainset

crf = torch.load('crf-0.pt')
inputs, target = trainset.__getitem__(0)
mask, hmap, target = torch.unsqueeze(inputs[0],0),torch.unsqueeze(inputs[1],0),torch.unsqueeze(target,0)
output = crf(mask.to('cuda:0').float(), hmap.to('cuda:0').float())
print(type(output))
output = output[0,0,:,:].cpu().detach().numpy()
print(np.max(output), np.min(output), np.mean(output))
a = np.uint8(output < 0.39)
cv2.imwrite('out-0.jpg', a)
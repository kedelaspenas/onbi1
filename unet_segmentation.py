import torch
from unet.unet import UNet
from yolocrf import train_files, test_files
import numpy as np, os, cv2

net = UNet(n_channels=1, n_classes=1).cuda()
net.load_state_dict(torch.load('CP64.pth', map_location={'cuda:1':'cuda:0'}))

os.makedirs('intermediate2/train', exist_ok=True)
os.makedirs('intermediate2/test', exist_ok=True)

for idx, dataset in enumerate([train_files, test_files]):
	for file in dataset:
		filename = file.split('/')[-1][:-4]
		img_orig = np.expand_dims(cv2.imread(file,0),0).astype(np.float32)
		pred = net(torch.from_numpy(np.expand_dims(img_orig,0)).cuda())
		output = np.squeeze(pred.detach().cpu().numpy())
		thresh = np.array(output > output.mean(), dtype=np.uint8)
		thresh = cv2.resize(thresh, (416,416))
		ret, label = cv2.connectedComponents(thresh)
		np.savez(os.path.join('intermediate2', 'train' if idx==0 else 'test', filename + '.npz'), semantic=output, instance=label)
		cv2.imwrite(os.path.join('intermediate2', 'train' if idx==0 else 'test', filename + '-semantic.jpg'), output*255/output.max())
		cv2.imwrite(os.path.join('intermediate2', 'train' if idx==0 else 'test', filename + '-instance.jpg'), label*255/(label.max()-1))

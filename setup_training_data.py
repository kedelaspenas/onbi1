import os, cv2, numpy as np, shutil

path = '/home/kristofer/Downloads/stage1_train'
d = os.listdir(path)

np.random.shuffle(d)

os.makedirs('train/image')
os.makedirs('test/image')
os.makedirs('train/label')
os.makedirs('test/label')
os.makedirs('train/mask')
os.makedirs('test/mask')

def get_coords(f,img):
	h,w = img.shape[:2]
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
	for idx in range(1,num_labels):
		out = [0, centroids[idx][0]/w, centroids[idx][1]/h, stats[idx][cv2.CC_STAT_WIDTH]/w, stats[idx][cv2.CC_STAT_HEIGHT]/h]
		f.write(' '.join([str(i) for i in out]))

n_train = int(0.60*len(d))
n_test = len(d) - n_train
train_files = [os.path.join(path, i, 'images', i + '.png') for i in d[:n_train]]
train_files_seg = [os.path.join(path, i, 'masks', 'mask.png') for i in d[:n_train]]
for i in d[:n_train]:
	f = open(os.path.join('train','label',i+'.txt'), 'w')
	shutil.copy2(os.path.join(path, i, 'images', i + '.png'), 'train/image/')
	masks = [i for i in os.listdir(os.path.join(path, i, 'masks')) if 'mask' not in i]
	mask = cv2.imread(os.path.join(path, i, 'masks', masks[0]), 0)
	get_coords(f, mask)
	for m in masks[1:]:
		f.write('\n')
		temp = cv2.imread(os.path.join(path, i, 'masks', m), 0)
		get_coords(f, temp)
		mask = mask + temp
	f.close()
	cv2.imwrite(os.path.join('train', 'mask', i + '.png'), mask)
test_files = [os.path.join(path, i, 'images', i + '.png') for i in d[n_train:]]
test_files_seg = [os.path.join(path, i, 'masks', 'mask.png') for i in d[n_train:]]
for i in d[n_train:]:
	f = open(os.path.join('test','label',i+'.txt'), 'w')
	shutil.copy2(os.path.join(path, i, 'images', i + '.png'), 'test/image/')
	masks = [i for i in os.listdir(os.path.join(path, i, 'masks')) if 'mask' not in i]
	mask = cv2.imread(os.path.join(path, i, 'masks', masks[0]), 0)
	get_coords(f, mask)
	for m in masks[1:]:
		f.write('\n')
		temp = cv2.imread(os.path.join(path, i, 'masks', m), 0)
		get_coords(f, temp)
		mask = mask + temp
	f.close()
	cv2.imwrite(os.path.join('test', 'mask', i + '.png'), mask)

np.save('trn_imglist.npy', train_files)
np.save('trn_seglist.npy', train_files_seg)
np.save('val_imglist.npy', test_files)
np.save('val_seglist.npy', test_files_seg)
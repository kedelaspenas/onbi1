from keras.models import Sequential, Model
from keras.layers import UpSampling2D, Concatenate
import keras.backend as K
import tensorflow as tf
import os, cv2, numpy as np
from keras_yolov2_neuroblastoma import TinyYolo, get_mask, get_prediction, yolo_parameter
from yolov2_utils import decode_netout

model = TinyYolo('darknet_neuroblastoma.npy')
print(model.output)
pool1 = model.get_layer('layer2-maxpool')
up1 = UpSampling2D()(pool1.output)
pool2 = model.get_layer('layer4-maxpool')
up2 = UpSampling2D()(pool2.output)
up2 = UpSampling2D()(up2)
pool3 = model.get_layer('layer6-maxpool')
up3 = UpSampling2D()(pool3.output)
up3 = UpSampling2D()(up3)
up3 = UpSampling2D()(up3)
pool4 = model.get_layer('layer8-maxpool')
up4 = UpSampling2D()(pool4.output)
up4 = UpSampling2D()(up4)
up4 = UpSampling2D()(up4)
up4 = UpSampling2D()(up4)

concat = Concatenate()([up1, up2, up3, up4])
print(concat)

extractor_model = Model(inputs=model.inputs, outputs=[concat, model.outputs[0]])
extractor_model.summary()

def read_dataset(filename):
    f = open(filename)
    files = [i.strip() for i in f.readlines()]
    f.close()
    return files

train_files = read_dataset('train_neuroblastoma.txt')
test_files = read_dataset('test_neuroblastoma.txt')

for idx, fileset in enumerate([train_files,test_files]):
    for file in fileset:
        name = file.split('/')[-1][:-4]
        print(name)
        # img = cv2.imread('/home/imaging/kdelasp/dourflow/train/image' + name + '.png')
        img_orig = cv2.imread(file)
        img = cv2.resize(img_orig, (416, 416))
        img = img / 255.
        img = img[:,:,::-1]
        img_b = np.expand_dims(img, 0)
        out = extractor_model.predict(img_b)
        boxes = decode_netout(out[1][0], yolo_parameter()[0], 1)
        img_out = get_mask(img_orig,boxes,1.0,True,(416,416),show_score=True)
        img_out2 = get_prediction(img_orig,boxes)
        cv2.imwrite(os.path.join('intermediate', 'train' if idx==0 else 'test', name + '.png'), img_out2)
        np.savez_compressed(os.path.join('intermediate', 'train' if idx==0 else 'test', name), heatmaps=out[0][0], mask=img_out)

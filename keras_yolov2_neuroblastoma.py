import keras
from keras.models import Model
from keras import layers
import keras.backend as K
import numpy as np
from keras.layers.core import Lambda
import tensorflow as tf
from yolov2_utils import decode_netout, draw_boxes
import cv2

weights_dict = dict()
def load_weights_from_file(weight_file):
    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def set_layer_weights(model, weights_dict):
    for layer in model.layers:
        if layer.name in weights_dict:
            cur_dict = weights_dict[layer.name]
            current_layer_parameters = list()
            if layer.__class__.__name__ == "BatchNormalization":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
                current_layer_parameters.extend([cur_dict['mean'], cur_dict['var']])
            elif layer.__class__.__name__ == "Scale":
                if 'scale' in cur_dict:
                    current_layer_parameters.append(cur_dict['scale'])
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            elif layer.__class__.__name__ == "SeparableConv2D":
                current_layer_parameters = [cur_dict['depthwise_filter'], cur_dict['pointwise_filter']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            elif layer.__class__.__name__ == "Embedding":
                current_layer_parameters.append(cur_dict['weights'])
            else:
                # rot weights
                current_layer_parameters = [cur_dict['weights']]
                if 'bias' in cur_dict:
                    current_layer_parameters.append(cur_dict['bias'])
            model.get_layer(layer.name).set_weights(current_layer_parameters)

    return model


def TinyYolo(weight_file = None):
    global weights_dict
    weights_dict = load_weights_from_file(weight_file) if not weight_file == None else None
        
    dk_Input        = layers.Input(name = 'dk_Input', shape = (416, 416, 3,) )
    layer1_conv     = convolution(weights_dict, name='layer1-conv', input=dk_Input, group=1, conv_type='layers.Conv2D', filters=16, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=False)
    layer1_bn       = layers.BatchNormalization(name = 'layer1-bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(layer1_conv)
    layer1_act      = layers.LeakyReLU(name='layer1-act', alpha = 0.10000000149011612)(layer1_bn)
    layer2_maxpool  = layers.MaxPooling2D(name = 'layer2-maxpool', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(layer1_act)
    layer3_conv     = convolution(weights_dict, name='layer3-conv', input=layer2_maxpool, group=1, conv_type='layers.Conv2D', filters=32, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=False)
    layer3_bn       = layers.BatchNormalization(name = 'layer3-bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(layer3_conv)
    layer3_act      = layers.LeakyReLU(name='layer3-act', alpha = 0.10000000149011612)(layer3_bn)
    layer4_maxpool  = layers.MaxPooling2D(name = 'layer4-maxpool', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(layer3_act)
    layer5_conv     = convolution(weights_dict, name='layer5-conv', input=layer4_maxpool, group=1, conv_type='layers.Conv2D', filters=64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=False)
    layer5_bn       = layers.BatchNormalization(name = 'layer5-bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(layer5_conv)
    layer5_act      = layers.LeakyReLU(name='layer5-act', alpha = 0.10000000149011612)(layer5_bn)
    layer6_maxpool  = layers.MaxPooling2D(name = 'layer6-maxpool', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(layer5_act)
    layer7_conv     = convolution(weights_dict, name='layer7-conv', input=layer6_maxpool, group=1, conv_type='layers.Conv2D', filters=128, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=False)
    layer7_bn       = layers.BatchNormalization(name = 'layer7-bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(layer7_conv)
    layer7_act      = layers.LeakyReLU(name='layer7-act', alpha = 0.10000000149011612)(layer7_bn)
    layer8_maxpool  = layers.MaxPooling2D(name = 'layer8-maxpool', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(layer7_act)
    layer9_conv     = convolution(weights_dict, name='layer9-conv', input=layer8_maxpool, group=1, conv_type='layers.Conv2D', filters=256, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=False)
    layer9_bn       = layers.BatchNormalization(name = 'layer9-bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(layer9_conv)
    layer9_act      = layers.LeakyReLU(name='layer9-act', alpha = 0.10000000149011612)(layer9_bn)
    layer10_maxpool = layers.MaxPooling2D(name = 'layer10-maxpool', pool_size = (2, 2), strides = (2, 2), padding = 'valid')(layer9_act)
    layer11_conv    = convolution(weights_dict, name='layer11-conv', input=layer10_maxpool, group=1, conv_type='layers.Conv2D', filters=512, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=False)
    layer11_bn      = layers.BatchNormalization(name = 'layer11-bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(layer11_conv)
    layer11_act     = layers.LeakyReLU(name='layer11-act', alpha = 0.10000000149011612)(layer11_bn)
    layer12_maxpool = layers.MaxPooling2D(name = 'layer12-maxpool', pool_size = (2, 2), strides = (1, 1), padding = 'same')(layer11_act)
    layer13_conv    = convolution(weights_dict, name='layer13-conv', input=layer12_maxpool, group=1, conv_type='layers.Conv2D', filters=1024, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=False)
    layer13_bn      = layers.BatchNormalization(name = 'layer13-bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(layer13_conv)
    layer13_act     = layers.LeakyReLU(name='layer13-act', alpha = 0.10000000149011612)(layer13_bn)
    layer14_conv    = convolution(weights_dict, name='layer14-conv', input=layer13_act, group=1, conv_type='layers.Conv2D', filters=1024, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=False)
    layer14_bn      = layers.BatchNormalization(name = 'layer14-bn', axis = -1, epsilon = 9.999999747378752e-06, center = True, scale = True)(layer14_conv)
    layer14_act     = layers.LeakyReLU(name='layer14-act', alpha = 0.10000000149011612)(layer14_bn)
    layer15_conv    = convolution(weights_dict, name='layer15-conv', input=layer14_act, group=1, conv_type='layers.Conv2D', filters=30, kernel_size=(1, 1), strides=(1, 1), dilation_rate=(1, 1), padding='same', use_bias=True)
    layer16_region  = layer15_conv
    output = layers.Reshape((13, 13, 5, 4 + 1 + 1))(layer16_region)
    model           = Model(inputs = [dk_Input], outputs = [output])
    set_layer_weights(model, weights_dict)
    return model

def yolo_parameter():
    return [[0.7387679815292358, 0.8749459981918335, 2.4220399856567383, 2.6570401191711426, 4.3097100257873535, 7.0449299812316895, 10.246000289916992, 4.594279766082764, 12.686800003051758, 11.874099731445312], 1, 0.6000000238418579, 1, 1, 0.20000000298023224, 5, 1.0, 4, 1, 1, 1, 5, 1, 1]


def convolution(weights_dict, name, input, group, conv_type, filters=None, **kwargs):
    if not conv_type.startswith('layer'):
        layer = keras.applications.mobilenet.DepthwiseConv2D(name=name, **kwargs)(input)
        return layer
    elif conv_type == 'layers.DepthwiseConv2D':
        layer = layers.DepthwiseConv2D(name=name, **kwargs)(input)
        return layer
    
    inp_filters = K.int_shape(input)[-1]
    inp_grouped_channels = int(inp_filters / group)
    out_grouped_channels = int(filters / group)
    group_list = []
    if group == 1:
        func = getattr(layers, conv_type.split('.')[-1])
        layer = func(name = name, filters = filters, **kwargs)(input)
        return layer
    weight_groups = list()
    if not weights_dict == None:
        w = np.array(weights_dict[name]['weights'])
        weight_groups = np.split(w, indices_or_sections=group, axis=-1)
    for c in range(group):
        x = layers.Lambda(lambda z: z[..., c * inp_grouped_channels:(c + 1) * inp_grouped_channels])(input)
        x = layers.Conv2D(name=name + "_" + str(c), filters=out_grouped_channels, **kwargs)(x)
        weights_dict[name + "_" + str(c)] = dict()
        weights_dict[name + "_" + str(c)]['weights'] = weight_groups[c]
        group_list.append(x)
    layer = layers.concatenate(group_list, axis = -1)
    if 'bias' in weights_dict[name]:
        b = K.variable(weights_dict[name]['bias'], name = name + "_bias")
        layer = layer + b
    return layer


def get_prediction(img_or, boxes):
    return draw_boxes(img_or, boxes, ['neuroblastoma'])

def get_mask(img_or, boxes, percent=1.0):
    mask = np.zeros(img_or.shape)
    h,w,_ = img_or.shape
    for box in boxes:
        dx = (1-percent)/2*(box.xmax - box.xmin)
        dy = (1-percent)/2*(box.ymax - box.ymin)
        xmin = int((box.xmin + dx)*w)
        ymin = int((box.ymin + dy)*h)
        xmax = int((box.xmax - dx)*w)
        ymax = int((box.ymax - dy)*h)

        cv2.rectangle(mask, (xmin,ymin), (xmax,ymax), (255,255,255), -1)
    return mask

if __name__ == "__main__":
    get_mask('/home/kristofer/Documents/stuff/yolov2/data/neuroblastoma_phal_class/2018/JPEGImages/110084.jpg')
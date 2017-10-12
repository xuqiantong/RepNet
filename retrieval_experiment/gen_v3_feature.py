# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'caffe', 'python'))
import caffe
import numpy as np

parser = argparse.ArgumentParser(description='Generate and store GoogLeNet features of all images under a specified directory')
parser.add_argument('--gpu_id', dest='gpu_id',
                    help='GPU device to use [0]',
                    default=0, type=int)
parser.add_argument('--img_dir', dest='img_dir',
                    help='image dir',
                    default='', type=str)
parser.add_argument('--img_list', dest='img_list',
                    help='image list',
                    default='', type=str)
parser.add_argument('--feature_path', dest='feature_path',
                    help='feature path',
                    default='', type=str)
parser.add_argument('--net_def', dest='net_def',
                    help='network definition',
                    default='', type=str)
parser.add_argument('--weights', dest='weights',
                    help='pretrained weights',
                    default='', type=str)
#parser.add_argument('--feat_pool5', dest='feat_pool5',
#                    help='layer after conv5',
#                    default='pool5', type=str)
#parser.add_argument('--feat_fc6', dest='feat_fc6',
#                    help='layer after pool5',
#                    default='fc6', type=str)
#parser.add_argument('--feat_fc7', dest='feat_fc7',
#                    help='layer after fc6',
#                    default='fc6', type=str)
parser.add_argument('--feat', dest='feat',
                    help='layer for feature',
                    default='fc8_hash', type=str)
parser.add_argument('--feat_cc', dest='feat_cc',
                    help='layer for feature',
                    default='fc8_hash', type=str)
parser.add_argument('--fc_model', dest='fc_model',
                    help='fc layer',
                    default='prob', type=str)
parser.add_argument('--fc_color', dest='fc_color',
                    help='fc layer',
                    default='prob_2', type=str)
args = parser.parse_args()

#  设置gpu环境，载入网络模型
caffe.set_device(args.gpu_id)
caffe.set_mode_gpu()
net = caffe.Net(args.net_def,
                args.weights,
                caffe.TEST)
print '\nLoaded network {:s}'.format(args.weights)

#  transformer 为读图用的预处理工具
in_ = net.inputs[0]
in_shape = net.blobs[in_].data.shape
transformer = caffe.io.Transformer({in_: in_shape})
transformer.set_transpose(in_, (2, 0, 1))
transformer.set_raw_scale(in_, 255)
transformer.set_channel_swap(in_, (2, 1, 0))

counter = 0
start = time.time()
with open(args.img_list, 'r') as imglist:
    with open(args.feature_path, 'w') as bcpath:
        for f in imglist:
            img_path = os.path.join(args.img_dir, f.strip())
            input_img = caffe.io.load_image(img_path)
            if input_img == None or input_img.sum()==0:
                continue
            input_ = transformer.preprocess(in_,
                                            caffe.io.resize_image(input_img,
                                                                (in_shape[2], in_shape[3])))
            out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})

            feat_model = net.blobs[args.fc_model].data[0].flatten()
            feat_color = net.blobs[args.fc_color].data[0].flatten()
            feat_cc = net.blobs[args.feat_cc].data[0].flatten() 
            feat = net.blobs[args.feat].data[0].flatten()

            bcpath.write('%(fn)s' % {"fn":f.strip()})

            for k in range(len(feat_model)):
                bcpath.write(' %f' % feat_model[k])
            for k in range(len(feat_color)):
                bcpath.write(' %f' % feat_color[k])
            for k in range(len(feat)):
                bcpath.write(' %f' % feat[k])
            for k in range(len(feat_cc)):
                bcpath.write(' %f' % feat_cc[k])
            bcpath.write('\n')
            counter = counter+1
            print 'writing %(counter)d %(fn)s' % {"counter":counter, "fn":f.strip()}
            
print('Feature extraction cost %.2f s' % (time.time()-start))

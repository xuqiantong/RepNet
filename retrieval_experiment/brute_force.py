import os
import sys
import argparse
import time
from numpy import *
from numpy.random import *
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, 'caffe', 'python'))
import caffe
#import cv2

parser = argparse.ArgumentParser(description='compute mean average precision')

parser.add_argument('--gpu_id', dest='gpu_id',
                    help='GPU device to use [0]',
                    default=0, type=int)
parser.add_argument('--query_dir', dest='query_dir',
                    help='query dir',
                    default='', type=str)
parser.add_argument('--query_list', dest='query_list',
                    help='query list',
                    default='', type=str)
parser.add_argument('--siyang_label', dest='siyang_label',
                    help='siyang label',
                    default='../query/cropped_label/', type=str)
parser.add_argument('--siyang_feat', dest='siyang_feat',
                    help='siyang feature data path',
                    default='', type=str)
parser.add_argument('--wendeng_feat', dest='wendeng_feat',
                    help='wendeng feature data path',
                    default='', type=str)
parser.add_argument('--begin_loc', dest='begin_loc',
                    help='beginning location of the feature',
                    default=0, type=int)
parser.add_argument('--end_loc', dest='end_loc',
                    help='ending location (not inclusiv) of the feature',
                    default=0, type=int)
parser.add_argument('--nn_number', dest='nn_number',
                    help='the number of returned nearest neighbors',
                    default=1000, type=int)
parser.add_argument('--net_def', dest='net_def',
                    help='network definition',
                    default='', type=str)
parser.add_argument('--weights', dest='weights',
                    help='pretrained weights',
                    default='', type=str)
parser.add_argument('--feat', dest='feat',
                    help='which layer to compare',
                    default='fc7_clf_sigmoid', type=str)
#parser.add_argument('--model_fc', dest='model_fc',
#                    help='fc layer',
#                    default='prob', type=str)
#parser.add_argument('--color_fc', dest='color_fc',
#                    help='fc layer',
#                    default='prob_2', type=str)

args = parser.parse_args()

#load data set
data_num = 222628 + 389316
data_dim = args.end_loc-args.begin_loc
dataset = zeros([data_num, data_dim], dtype=float32)
img_names = []
img_labels = []

print 'Loading siyang dataset'
start = time.time()
count = 0
siyang_label = {}
with open(args.siyang_label, 'r') as f:
    for line in f:
        sp = line[:-1].split()
        siyang_label[sp[0]] = sp[1]

with open(args.siyang_feat, 'r') as f:
    for line in f:
        ln = line.split()
        img_name = ln[0]
        img_names.append('/media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/' + img_name)
        img_labels.append(siyang_label[img_name])
        #for i in range(1024):
        #    dataset[count][i] = ln[i+1]
        dataset[count] = ln[args.begin_loc:args.end_loc]
        count = count+1
print('Done in %.2f s' % (time.time()-start))

print 'Loading wendeng dataset'
start = time.time()
with open(args.wendeng_feat, 'r') as f:
    for line in f:
        ln = line.split()
        img_names.append('/media/megatron-home/dwliang/data/wendeng_res/' + ln[0])
        img_labels.append(int(-1))
        #for i in range(1024):
        #    dataset[count][i] = ln[i+1]
        dataset[count] = ln[args.begin_loc:args.end_loc]
        count = count+1
        if count == data_num:
            break
print('Done in %.2f s' % (time.time()-start))

#load caffe model, extract features, and compute mAP
print 'Loading caffe model'
start = time.time()
caffe.set_device(args.gpu_id)
caffe.set_mode_gpu()
net = caffe.Net(args.net_def,
                args.weights,
                caffe.TEST)
print '\nLoaded network {:s}'.format(args.weights)
print('Done in %.2f s' % (time.time()-start))

print 'computing mean average precision'
start = time.time()
in_ = net.inputs[0]
in_shape = net.blobs[in_].data.shape
transformer = caffe.io.Transformer({in_: in_shape})
transformer.set_transpose(in_, (2, 0, 1))
transformer.set_raw_scale(in_, 255)
transformer.set_channel_swap(in_, (2, 1, 0))

mean_average_precision = 0.0
mean_query_time = 0.0
query_num = 0;
precsion_at_k = zeros([args.nn_number,], dtype=float32)

print 'computing dot product of dataset rows'
start = time.time()
dataset_dot_product = array([dataset[i].dot(dataset[i]) for i in range(data_num)])
print('Done in %.2f s' % (time.time()-start))
#temp = 0

des = open('asd.txt', "w")
with open(args.query_list, 'r') as f:
    for line in f:
        img_name = line.split()[0]
        img_label = int(line.split()[1])
        img_path = os.path.join(args.query_dir, img_name)
        input_img = caffe.io.load_image(img_path)
        start = time.time()
        if input_img == None or input_img.sum()==0:
            print 'bad image'
            continue
        else:
            input_ = transformer.preprocess(in_,
                                            caffe.io.resize_image(input_img,
                                                                (in_shape[2], in_shape[3])))
            out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})
            feat = net.blobs[args.feat].data[0].flatten()
            #feat = [int(i>0.5) for i in feat]
            #binary_code = 0
            #for i in range(len(feat)):
            #    binary_code = (binary_code|(1 if feat[i]>=0.5 else 0))<<1
            #binary_code = binary_code>>1
        print('extracting features for %s in %.2f s' % (img_name, time.time()-start))

        #query and show the designated number of returned images
        query_num += 1

        start = time.time()
        dist = dataset_dot_product-2*dataset.dot(feat)
        result = dist.argsort()
        mean_query_time += time.time()-start
        print('querying %s done in %.2f s' % (img_name, time.time()-start))

        #compute average precision & precision at k
        match_num = 0;
        average_precision = 0.0
#        temp = temp + 1
#        show_result = open('show_result_id/' + str(temp) + '.txt','w')
#        show_result.write('/media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/' + img_name);
#        show_result.write('\n');
        for i in range(args.nn_number):
#            show_result.write(img_names[result[i]] + '\n')
            if int(img_labels[result[i]]) == img_label:
                match_num += 1
                average_precision += float(match_num) / float(i+1)
            precsion_at_k[i] += float(match_num)/float(i+1)

        gt_num = sum(array(img_labels) == img_label)
        average_precision = average_precision / float(gt_num)
        if average_precision > 0.95:
            des.write(img_path + '\n')
            for i in range(10):
                des.write(img_names[result[i]])
            des.write('*************\n')
        mean_average_precision += average_precision
        print('ground truth number is %d' % int(gt_num))
        print('average precision for %s is %.2f\n' % (img_name, average_precision))
#        show_result.close()

# compute mean average precision and mean query time
mean_average_precision /= query_num
mean_query_time /= query_num

print('mean average precision is %.2f' % (mean_average_precision))
print('mean query time is %.2f s' % mean_query_time)

# save precision@k
with open('/home/xqt/essence/V0_fc7_3_precision_at_k.txt', 'w') as f:
    for i in range(args.nn_number):
        f.write('%f\n' % (precsion_at_k[i]/query_num))

with open('/home/xqt/essence/V0_fc7_3_map.txt', 'w') as f:
    f.write('map=%f, mean_query_time=%f' % (mean_average_precision, mean_query_time))

#query_img = cv2.imread(img_path)
#cv2.imshow('query', query_img)

#for i in range(len(result[0])):
#    res_id = int(result[0][i])
#    res_path = os.path.join(args.img_dir, img_names[res_id])
#    res_img = cv2.imread(res_path)
#    cv2.imshow('result', res_img)
#    key = cv2.waitKey(0)
#    if key == 27:
#        break


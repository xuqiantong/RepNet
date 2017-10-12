import os
import sys
import argparse
import time
import numpy as np
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
parser.add_argument('--fc_model', dest='fc_model',
                    help='fc layer',
                    default='prob', type=str)
parser.add_argument('--fc_color', dest='fc_color',
                    help='fc layer',
                    default='prob_2', type=str)
parser.add_argument('--mAP_path', dest='mAP_path',
                    help='mAP path',
                    default='', type=str)
parser.add_argument('--p@k_path', dest='p_k_path',
                    help='p@k_path',
                    default='', type=str)

args = parser.parse_args()

#load data set
data_num = 222628 + 389316
data_dim = args.end_loc-args.begin_loc
dataset = np.zeros([data_num, data_dim], dtype=np.float32)
img_names = []
img_labels = []
buckets = {}

def form_key(clazz, color):
    return str(clazz) + '_' + str(color)

print 'Loading siyang dataset'
start = time.time()
count = 0
siyang_label = {}
with open(args.siyang_label, 'r') as f:
    for line in f:
        line = line.strip()
        sp = line.split()
        siyang_label[sp[0]] = sp[1]

color_buc_num = 2
model_buc_num = 2
with open(args.siyang_feat, 'r') as f:
    for line in f:
        line = line.strip()
        ln = line.split()
        img_name = ln[0]
        img_names.append('/media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/' + img_name)
        img_labels.append(siyang_label[img_name])
        f_model = np.array([float(i) for i in ln[1:251]]).argsort()
        f_color = np.array([float(i) for i in ln[251:258]]).argsort()
        for i in range(color_buc_num):
            color = f_color[-1-i]
            for j in range(model_buc_num):
                model = f_model[-1-j]
                key = form_key(model, color)
                if not buckets.has_key(key):
                    buckets[key] = []
                buckets[key].append(count)
        #for i in range(1024):
        #    dataset[count][i] = ln[i+1]
        dataset[count] = ln[args.begin_loc:args.end_loc]
        count = count+1
print('Done in %.2f s' % (time.time()-start))

print 'Loading wendeng dataset'
start = time.time()
with open(args.wendeng_feat, 'r') as f:
    for line in f:
        line = line.strip()
        ln = line.split()
        img_names.append('/media/megatron-home/dwliang/data/wendeng_res/' + ln[0])
        img_labels.append(int(-1))
        f_model = np.array([float(i) for i in ln[1:251]]).argsort()
        f_color = np.array([float(i) for i in ln[251:258]]).argsort()
        for i in range(color_buc_num):
            color = f_color[-1-i]
            for j in range(model_buc_num):
                model = f_model[-1-j]
                key = form_key(model, color)
                if not buckets.has_key(key):
                    buckets[key] = []
                buckets[key].append(count)
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

print 'Computing mean average precision\n'
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
precsion_at_k = np.zeros([args.nn_number,], dtype=np.float32)

print 'Computing dot product of dataset rows\n'
start = time.time()
dataset_dot_product = np.array([dataset[i].dot(dataset[i]) for i in range(data_num)])
print('Done in %.2f s' % (time.time()-start))
#temp = 0

#des = open('asd.txt', "w")
with open(args.query_list, 'r') as f:
    for line in f:
        img_name = line.split()[0]
        #if not img_name == '0090436.jpg':
        #    continue
        img_label = line.split()[1]
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
            #print feat[0:10]
            feat_model = net.blobs[args.fc_model].data[0].flatten()
            #print feat_model[0:10]
            #tmp = list(feat_model)
            #label_model = tmp.index(max(tmp))
            label_model = np.where(feat_model == max(feat_model))[0][0]
            feat_color = net.blobs[args.fc_color].data[0].flatten()
            #print feat_color[0:7]
            #tmp = list(feat_color)
            #label_color = tmp.index(max(tmp))
            label_color = np.where(feat_color == max(feat_color))[0][0]
            #feat = [int(i>0.5) for i in feat]
            #binary_code = 0
            #for i in range(len(feat)):
            #    binary_code = (binary_code|(1 if feat[i]>=0.5 else 0))<<1
            #binary_code = binary_code>>1
        print('extracting features for %s in %.2f s' % (img_name, time.time()-start))

        #query and show the designated number of returned images
        query_num += 1

        #extract data in the same bucket
        bucket_list = buckets[form_key(label_model, label_color)]
        #print form_key(label_model, label_color)
        #print "%d in current bucket." % len(bucket_list)
        data = np.array([dataset[i] for i in bucket_list])
        data_dot_product = np.array([dataset_dot_product[i] for i in bucket_list])

        start = time.time()
        dist = data_dot_product-2*data.dot(feat)
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
        #des.write("Query: " + img_name + '\n') 
        #for i in range(min(len(bucket_list), args.nn_number)):
          #  des.write(img_names[bucket_list[result[i]]] + '\n')
        for i in range(min(len(bucket_list), args.nn_number)):
#            show_result.write(img_names[result[i]] + '\n')
            if img_labels[bucket_list[result[i]]] == img_label:
                match_num += 1
                average_precision += float(match_num) / float(i+1)
            precsion_at_k[i] += float(match_num)/float(i+1)

        gt_num = sum(np.array(img_labels) == img_label)
        average_precision = average_precision / float(gt_num)
        #if average_precision > 0.95:
        #    des.write(img_path + '\n')
        #    for i in range(10):
        #        des.write(img_names[result[i]])
        #    des.write('*************\n')
        mean_average_precision += average_precision
        print('ground truth number is %d' % int(gt_num))
        print('average precision for %s is %.2f\n' % (img_name, average_precision))
#        show_result.close()
        #break
# compute mean average precision and mean query time
mean_average_precision /= query_num
mean_query_time /= query_num

print('mean average precision is %.2f' % (mean_average_precision))
print('mean query time is %.2f s' % mean_query_time)

# save precision@k
with open(args.p_k_path, 'w') as f:
    for i in range(args.nn_number):
        f.write('%f\n' % (precsion_at_k[i]/query_num))

with open(args.mAP_path, 'w') as f:
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


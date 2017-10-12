import os
import sys
import argparse
import time
import numpy as np
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, 'caffe', 'python'))
import caffe
import matplotlib.pyplot as plt
from scipy import misc

caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Net('/home/xqt/exp/deploy_v2.prototxt',
                '/home/xqt/exp_res/v2_iter_160000.caffemodel',
                caffe.TEST)
print '\nLoaded network {:s}'.format('/home/xqt/bishe/exp_res/v2_cmp_iter_92593.caffemodel')

in_ = net.inputs[0]
in_shape = net.blobs[in_].data.shape
transformer = caffe.io.Transformer({in_: in_shape})
transformer.set_transpose(in_, (2, 0, 1))
transformer.set_raw_scale(in_, 255)
transformer.set_channel_swap(in_, (2, 1, 0))

img_path = '/media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/0141295.jpg'
input_img = caffe.io.load_image(img_path)
input_ = transformer.preprocess(in_, caffe.io.resize_image(input_img, (in_shape[2], in_shape[3])))
out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})
f_base = net.blobs['fc6'].data[0].flatten()
#f_cc = net.blobs['fc7_cc'].data[0].flatten()
f_color = net.blobs['fc8_color'].data[0].flatten()
f_class = net.blobs['fc8_class'].data[0].flatten()
f_triplet = net.blobs['fc9_triplet'].data[0].flatten()
#f_id = net.blobs['fc10_id'].data[0].flatten()

org_img = misc.imread(img_path)
[h, w, c] = org_img.shape
r_base = np.zeros([h,w], dtype=np.float32)
r_color = np.zeros([h,w], dtype=np.float32)
r_class = np.zeros([h,w], dtype=np.float32)
#r_cc = np.zeros([h,w], dtype=np.float32)
r_triplet = np.zeros([h,w], dtype=np.float32)
#r_id = np.zeros([h,w], dtype=np.float32)
cnt = np.zeros([h,w], dtype=np.float32)

print "here"

D = 40
stride = 5

for i in range(0, h-D, stride):
	for j in range(0, w-D, stride):
		print i, j
		out_img = org_img.copy()
		out_img[i:i+D, j:j+D, :] = 0
		misc.imsave('tmp.jpg', out_img)
		input_img = caffe.io.load_image('tmp.jpg')
		input_ = transformer.preprocess(in_, caffe.io.resize_image(input_img, (in_shape[2], in_shape[3])))
		out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})
		f0 = net.blobs['fc6'].data[0].flatten()
		f1 = net.blobs['fc8_color'].data[0].flatten()
		f2 = net.blobs['fc8_class'].data[0].flatten()
		f3 = net.blobs['fc9_triplet'].data[0].flatten()
		#f4 = net.blobs['fc7_cc'].data[0].flatten()
		#f4 = net.blobs['fc10_id'].data[0].flatten()

		d_base = np.linalg.norm(f0-f_base)
		d_color = np.linalg.norm(f1-f_color)
		d_class = np.linalg.norm(f2-f_class)
		d_triplet = np.linalg.norm(f3-f_triplet)
		#d_id = np.linalg.norm(f4-f_id)
		#d_cc = np.linalg.norm(f4-f_cc)

		cnt[i:i+D, j:j+D] += 1
		r_base[i:i+D, j:j+D] += d_base
		r_color[i:i+D, j:j+D] += d_color
		r_class[i:i+D, j:j+D] += d_class
		r_triplet[i:i+D, j:j+D] += d_triplet
		#r_id[i:i+D, j:j+D] += d_id
		#r_cc[i:i+D, j:j+D] += d_cc

for i in range(h):
	for j in range(w):
		if cnt[i, j] == 0:
			cnt[i, j] = 1
r_base = np.divide(r_base, cnt)
r_base = np.array(r_base /np.max(r_base)* 255, dtype=np.int32)
r_color = np.divide(r_color, cnt)
r_color = np.array(r_color /np.max(r_color)* 255, dtype=np.int32)
r_class = np.divide(r_class, cnt)
r_class = np.array(r_class /np.max(r_class)* 255, dtype=np.int32)
r_triplet = np.divide(r_triplet, cnt)
r_triplet = np.array(r_triplet /np.max(r_triplet)* 255, dtype=np.int32)
#r_cc = np.divide(r_cc, cnt)
#r_cc = np.array(r_cc /np.max(r_cc)* 255, dtype=np.int32)
#r_id = np.divide(r_id, cnt)
#r_id = np.array(r_id /np.max(r_id)* 255, dtype=np.int32)

misc.imsave('r_base.jpg', r_base)
misc.imsave('r_color.jpg', r_color)
misc.imsave('r_class.jpg', r_class)
misc.imsave('r_triplet.jpg', r_triplet)
#misc.imsave('r_id.jpg', r_id)
#misc.imsave('r_cc.jpg', r_cc)






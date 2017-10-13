import os
import sys
import time
import argparse
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'caffe', 'python'))
import caffe
import numpy as np
from scipy import spatial

caffe.set_device(0) 
caffe.set_mode_gpu()
net = caffe.Net('/home/xqt/exp/deploy_v5.prototxt', 
                # '/home/xqt/exp/deploy_v2_cmp.prototxt'
                # '/home/xqt/exp/deploy_v6.prototxt'
                #  deploy_v5
                '/home/xqt/exp_res/v5_iter_175610.caffemodel', 
                # '/home/xqt/essence/v2_cmp_iter_203510.caffemodel'
                # '/home/xqt/exp_res/v6_iter_296864.caffemodel'
                # v5_iter_175610
                caffe.TEST)
print '\nLoaded network {:s}'.format('v6_iter_296864.caffemodel')

in_ = net.inputs[0]
in_shape = net.blobs[in_].data.shape
transformer = caffe.io.Transformer({in_: in_shape})
transformer.set_transpose(in_, (2, 0, 1))
transformer.set_raw_scale(in_, 255)
transformer.set_channel_swap(in_, (2, 1, 0))

f_sls1 = np.zeros([78982, 2048], dtype=np.float32)
f_sls2 = np.zeros([78982, 1000], dtype=np.float32)
cnt = 0

img_dir = '/media/mmr6-home/lhy/Documents/Data/Vehicles/vehicles_with_plate/cropped_uncovered/'
out_path = '/home/xqt/asd/cos_dis_v6.dis'
with open('/home/xqt/exp/list.txt', 'r') as imglist:
    with open(out_path, 'w') as bcpath:
        for f in imglist:
            img_path = os.path.join(img_dir, f.strip())
            input_img = caffe.io.load_image(img_path)
            if input_img == None or input_img.sum()==0:
                print 'fuck\n'
                continue
            input_ = transformer.preprocess(in_,
                                            caffe.io.resize_image(input_img,
                                                                (in_shape[2], in_shape[3])))
            out = net.forward_all(**{in_: input_.reshape((1, 3, in_shape[2], in_shape[3]))})

            f_sls1[cnt] = net.blobs['fc7_org'].data[0].flatten() 
            f_sls2[cnt] = net.blobs['fc8_triplet'].data[0].flatten()
            
            #cos_dis = spatial.distance.cosine(feat_cc, feat)
            #bcpath.write(str(cos_dis) + '\n')

            cnt = cnt+1
            if cnt%100 == 0:
                print 'writing %(counter)d %(fn)s' % {"counter":cnt, "fn":f.strip()}
            #print 'feat len: %d' % len(feat)
            #break
print('Feature extraction done')


from sklearn.cross_decomposition import CCA
from scipy.stats.stats import pearsonr
cca = CCA(n_components=1)
cca.fit(f_sls1, f_sls2)
X_c, Y_c = cca.transform(f_sls1, f_sls2)
X_c = X_c.reshape(78982).tolist()
Y_c = Y_c.reshape(78982).tolist()
d = pearsonr(X_c, Y_c)
print d


# v2_cmp, f_sls1 | f_sls2 : 0.99858308591425904
# v6,     f_sls1 | f_sls2 : 0.98989084710623576
# v5,     f_sls1 | f_sls2 : 0.99750778616168401
# v2,     f_sls1 | f_sls2 : 0.99533002790822001

# v2_cmp, f_acs | f_sls2  : 0.99428906422804875
# v6,     f_acs | f_sls2  : 0.99079317473380413

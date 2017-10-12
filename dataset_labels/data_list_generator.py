import h5py
#import caffe
import numpy as np
from random import randint
import random

src = open('Training_data_labels.txt')
dic = {}
dic_test = {}
dic_train = {}
test_samples = [0]*250

#read & seperate data

des1 = open('train_list.txt', "w")
des2 = open('test_list.txt', "w")
lines = src.readlines()
for i,l in enumerate(lines):
    l = l[:-1]
    sp = l.split(' ')
    dic[sp[0]] = sp[1:5]
    test_samples[int(sp[2])] += 1

test_samples = [int(x/10) for x in test_samples]
print sum(x>0 for x in test_samples)

for k,v in dic.items():
    if test_samples[int(v[1])] > 0:
        test_samples[int(v[1])] -= 1
        dic_test[k] = v
    else:
        dic_train[k] = v

for k,v in dic_train.items():
    attr = ' '.join(v)
    des1.write(k + ' ' + attr +'\n')

for k,v in dic_test.items():
    attr = ' '.join(v)
    des2.write(k + ' ' + attr +'\n')

src.close()
des1.close()
des2.close()


# for multi-classification
'''
src1 = open('train_list.txt')
src2 = open('test_list.txt')
des1 = open('train_color.txt', "w")
des2 = open('train_class.txt', "w")
des3 = open('test_color.txt', "w")
des4 = open('test_class.txt', "w")

lines1 = src1.readlines()
lines2 = src2.readlines()

for l in lines1:
    sp = l[:-1].split(' ')
    des1.write(sp[0] + ' ' + sp[3] + '\n')
    des2.write(sp[0] + ' ' + sp[2] + '\n')

for l in lines2:
    sp = l[:-1].split(' ')
    des3.write(sp[0] + ' ' + sp[3] + '\n')
    des4.write(sp[0] + ' ' + sp[2] + '\n')

src1.close()
des1.close()
des2.close()
src2.close()
des3.close()
des4.close()
'''

# for triplet net
'''
# test data 
des1 = open('exp/v4/test_a_class.txt', "w")
des2 = open('exp/v4/test_p_color.txt', "w")
des3 = open('exp/v4/test_n_weight.txt', "w")

test_list = [k for k, v in dic_test.items()]
test_nb = {}

for k, v in dic_test.items():
    id = v[0]
    if not test_nb.has_key(id):
        test_nb[id] = []
    test_nb[id].append(k)

l = len(test_list)
w1 = []
w2 = []
w3 = []


for anchor in test_list:
    model = dic_test[anchor][1]
    color = dic_test[anchor][2]

    anchor_neighbors = test_nb[dic_test[anchor][0]]
    loc = anchor_neighbors.index(anchor)
    p = anchor_neighbors[(loc+1)%len(anchor_neighbors)]

    k = randint(0,l-1)
    while test_list[k] in anchor_neighbors:
        k = randint(0,l-1)
            
        
    w1.append(anchor + '.jpg ' + model + '\n')
    w2.append(p + '.jpg ' + color + '\n')
    w3.append(test_list[k] + '.jpg ' + '1\n')
    

order = [i for i in range(len(w1))]
random.shuffle(order)
for i in range(len(w1)):
    j = order[i]
    des1.write(w1[j])
    des2.write(w2[j])
    des3.write(w3[j])

des1.close()
des2.close()
des3.close()


#training data
des1 = open('exp/train_a_class.txt', "w")
des2 = open('exp/train_p_color.txt', "w")
des3 = open('exp/train_n_weight.txt', "w")

train_list = [k for k, v in dic_train.items()]
id_nb = {}
color_nb = {}
model_nb = {}

for k, v in dic_train.items():
    id = v[0]
    model = v[1]
    color = v[2]

    if not id_nb.has_key(id):
        id_nb[id] = []
    id_nb[id].append(k)

    if not color_nb.has_key(color):
        color_nb[color] = []
    color_nb[color].append(k)

    if not model_nb.has_key(model):
        model_nb[model] = []
    model_nb[model].append(k)

## 9543 different ids
## 522 id only appear once
## 977 id appear twice
## 1384 id appear three times

l = len(train_list)
w1 = []
w2 = []
w3 = []


i = 0
for anchor in train_list:
    i += 1
    print i

    id = dic_train[anchor][0]
    model = dic_train[anchor][1]
    color = dic_train[anchor][2]

    positive = []
    anchor_neighbors = id_nb[id]
    loc = anchor_neighbors.index(anchor)
    for j in range(1, min(8, len(anchor_neighbors))):
        positive.append(anchor_neighbors[(loc+j)%len(anchor_neighbors)])

    for p in positive:
        negative = []
        for j in range(10):
            k = randint(0,l-1)
            while train_list[k] in anchor_neighbors or train_list[k] in negative \
                    or train_list[k] in model_nb[model] \
                    or train_list[k] in color_nb[color]:
                k = randint(0,l-1)

            w1.append(anchor + '.jpg ' + model + '\n')
            w2.append(p + '.jpg ' + color + '\n')
            w3.append(train_list[k] + '.jpg ' + '3\n')

        #negative = [x for x in model_nb[dic_train[anchor][1]] if x not in anchor_neighbors and x not in color_nb[dic_train[anchor][2]]]
        negative = list(set(model_nb[model]) - set(color_nb[color]) - set(anchor_neighbors))
        random.shuffle(negative)
        for j in range(min(10, len(negative))):
            w1.append(anchor + '.jpg ' + model + '\n')
            w2.append(p + '.jpg ' + color + '\n')
            w3.append(negative[j] + '.jpg ' + '2\n')

        #negative = [x for x in model_nb[dic_train[anchor][1]] if x in color_nb[dic_train[anchor][2]] and x not in anchor_neighbors]
        negative = list(set(model_nb[model]) & set(color_nb[color]) - set(anchor_neighbors))
        random.shuffle(negative)
        for j in range(min(10, len(negative))):
            w1.append(anchor + '.jpg ' + model + '\n')
            w2.append(p + '.jpg ' + color + '\n')
            w3.append(negative[j] + '.jpg ' + '1\n')

    

order = [i for i in range(len(w1))]
random.shuffle(order)
for i in range(len(w1)):
    j = order[i]
    des1.write(w1[j])
    des2.write(w2[j])
    des3.write(w3[j])

des1.close()
des2.close()
des3.close()

'''


#hdf5
'''
SIZE = 224 # fixed size to all images
dir = '/media/megatron-home/lhy/Documents/Data/Vehicles/VehicleID/cropped/'
with open( 'label.txt', 'r' ) as T :
    lines = T.readlines()

#print lines
# If you do not have enough memory split data into
# multiple batches and generate multiple separate h5 files
X = np.zeros( (len(lines), 3, SIZE, SIZE), dtype='f4' )
y = np.zeros( (len(lines),2), dtype='f4' )
for i,l in enumerate(lines):
    l = l[:-1]
    print l
    sp = l.split(' ')
    print sp[1]
    #img = caffe.io.load_image( dir+sp[0]+'.jpg' )
    #img = caffe.io.resize( img, (3, SIZE, SIZE) ) # resize to fixed size

    #you may apply other input transformations here...
    #X[i] = img
    y[i][0] = int(sp[2])
    y[i][1] = int(sp[3])
    print X[i].size
    print y[i]
    break

with h5py.File('train.h5','w') as H:
        #print "hdf5 writing"
    H.create_dataset( 'X', data=X ) # note the name X given to the dataset!
    H.create_dataset( 'y', data=y ) # note the name y given to the dataset!
with open('train_h5_list.txt','w') as L:
    L.write( 'train.h5' ) # list all h5 files you are going to use
'''
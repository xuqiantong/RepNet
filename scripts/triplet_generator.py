import numpy as np
from random import randint
import random


def form_key(model, color, fb):
	return model + ' ' + color + ' ' + fb

def form_id(id, fb):
	return id_list.index(id + '_' + fb)


###

# Data seperating

###
src = open('Training_data_labels.txt')
des1 = open('train_list.txt', "w")
des2 = open('test_list.txt', "w")

lines = src.readlines()

##
## form new id
##
id_list = []
dic = {}
for line in lines:
    sp = line.split()
    if sp[4] == '0':
    	continue
    key = sp[1] + '_' + sp[4]
    if key in id_list:
        continue
    id_list.append(key)

    if not dic.has_key(sp[1]):
    	dic[sp[1]] = 0
    dic[sp[1]] += 1

print len(id_list)
print len(dic)

##
## seperate data
##

'''

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


###

# Triplet Generating

###

src1 = open('train_list.txt')
src2 = open('test_list.txt')
des1 = open('T-f_train_class.txt', "w")
des2 = open('T-f_train_color.txt', "w")
des3 = open('T-f_train_id.txt', "w")
dic_train = {}
dic_test = {}
dic = {}
w1 = []
w2 = []
w3 = []

lines = src1.readlines()
for l in lines:
	sp = l[:-1].split()
	if sp[4] == '0':
		continue
	dic_train[sp[0]] = sp[1:5]
	k = form_key(sp[2], sp[3], sp[4])
	if not dic.has_key(k):
		dic[k] = {}
	if not dic[k].has_key(sp[1]):
		dic[k][sp[1]] = []
	dic[k][sp[1]].append(sp[0])
print len(dic)


for key, group in dic.items():
	print key
	sp = key.split(' ')
	model = sp[0]
	color = sp[1]
	fb = sp[2]

	for k, v in group.items():
		id = form_id(k, fb)
		l = len(v)
		if l == 1:
			continue
		positive_pairs = []
		for i in range(l-1):
			for j in range(i+1, l):
				positive_pairs.append((v[i], v[j]))
		
		negtive_items = []
		random.seed()
		for kk, vv in group.items():
			if kk == k:
				continue
			negtive_items.append(vv[random.randint(0,len(vv)-1)])

		for pp in positive_pairs:
			for n in negtive_items:
				w1.append(pp[0] + ' ' + model + '\n')
				w2.append(pp[1] + ' ' + color + '\n')
				w3.append(n + ' ' + str(id) + '\n')
print len(w1)

order = [i for i in range(len(w1))]
random.shuffle(order)
for i in range(len(w1)):
    j = order[i]
    des1.write(w1[j])
    des2.write(w2[j])
    des3.write(w3[j])

'''
import numpy as np
from random import randint
import random

src1 = open('siyang_label.txt')
src2 = open('Training_data_labels.txt')
des = open('query_random_1000.txt', "w")

train_list = []
lines = src2.readlines()
for line in lines:
	sp = line.split()
	if sp[4] == '0':
		continue

	if sp[1] not in train_list:
		train_list.append(sp[1])

dic_valid = {}
name_valid = {}
lines = src1.readlines()
for line in lines:
	sp = line.split()
	name = sp[0]
	sp = sp[1].split('_')
	if sp[1] == '0' or sp[0] in train_list:
		continue

	if not dic_valid.has_key(sp[0]):
		dic_valid[sp[0]] = 0
		name_valid[sp[0]] = name
	dic_valid[sp[0]] += 1

res = []
for k,v in dic_valid.items():
	res.append((k,v))

res.sort(key=lambda tup: tup[1], reverse = True)

for i in range(1000):
	des.write(name_valid[res[i][0]] + ' ' + res[i][0] + '_1\n')
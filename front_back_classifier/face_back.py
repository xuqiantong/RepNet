from numpy import *

#FB label merger
'''
src1 = open('FB.fea')
src2 = open('label.txt')
dic = {}
lines = src2.readlines()
for l in lines:
	sp = l[:-1].split()
	dic[sp[0] + '.jpg'] = sp[1:4]

lines = src1.readlines()
img_fb = []

for l in lines:
	sp = l[:-1].split()
	if float(sp[1]) > float(sp[2]):
		label = 0
	else:
		label = 1
	img_fb.append((sp[0], label))
	dic[sp[0]].append(str(label))

des = open('label4.txt', "w")
for k, v in dic.items():
	des.write(k + ' ' + ' '.join(v) + '\n')
'''

#calculate intra-class distance
'''
src1 = open('label.txt')
src2 = open('../v1_siyang_fc7.fea')

vid = {}
vgroup = {}
lines = src1.readlines()
for l in lines:
	sp = l[:-1].split(' ')
	vid[sp[0]+'.jpg'] = sp[1]
	vgroup[sp[1]] = []
print len(vid)

lines = src2.readlines()
dataset = zeros([78982, 2048], dtype=int)
i = 0
for l in lines:
	sp = l[:-1].split(' ')
	if not vid.has_key(sp[0]):
		continue
	id = vid[sp[0]]
	vgroup[id].append((sp[0],i))
	dataset[i] = sp[1:2049]
	i += 1
print i

des1 = open('train_id_list.txt', "w")
des2 = open('train_id_matrx.txt', "w")
for k, v in vgroup.items():
	print k
	temp = [j[0] for j in v]
	ind = [j[1] for j in v]
	des1.write(k + '\t' + ' '.join(temp) + '\n')
	l = len(v)
	des2.write(k + ',' + str(l)+ '\n')
	if l < 2:
		des2.write('0\n')
		continue
	for i in range(1,l):
		v1 = dataset[i]
		for j in range(0,i):
			des2.write(str(sum(dataset[j]^v1)) + ' ')
		des2.write('\n')

	#break
'''



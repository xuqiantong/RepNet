import sys
import numpy as np
import re
import time
import difflib
import array
import requests
import os
import shutil
import scipy.spatial.distance as spd
import numpy as np
from numpy.random import *
from random import randint
import random
import matplotlib.pyplot as plt
from scipy import misc


src_m1 = open('../exp_res/res/pre-v4_2048_r1000_precision_at_k.txt')
src_m2 = open('../exp_res/res/pre-v7_6144_r1000_precision_at_k.txt')
src_m3 = open('../exp_res/res/V2_cmp_2348_r1000_precision_at_k.txt')
src_m4 = open('../exp_res/res/v2_2348_r1000_precision_at_k.txt')
src_m5 = open('../exp_res/res/V5_2348_r1000_precision_at_k.txt')
src_m6 = open('../exp_res/res/v6_2348_r1000_precision_at_k.txt')
src_m7 = open('../exp_res/res/cvpr_n_2048_r1000_precision_at_k.txt')
src_m4_b = open('../exp_res/res/v2_bucket2-2_r1000_precision_at_k.txt')
src_m5_b = open('../exp_res/res/v5_bucket2-2_r1000_precision_at_k.txt')
src_m6_b = open('../exp_res/res/v6_bucket2-2_r1000_precision_at_k.txt')
l = 100
x = [i for i in range(1,1+l)]
p_m1 = []
lines = src_m1.readlines()
for i in range(l):
	p_m1.append(float(lines[i]))

p_m2 = []
lines = src_m2.readlines()
for i in range(l):
	p_m2.append(float(lines[i]))

p_m3 = []
lines = src_m3.readlines()
for i in range(l):
	p_m3.append(float(lines[i]))

p_m4 = []
lines = src_m4.readlines()
for i in range(l):
	p_m4.append(float(lines[i]))

p_m5 = []
lines = src_m5.readlines()
for i in range(l):
	p_m5.append(float(lines[i]))

p_m6 = []
lines = src_m6.readlines()
for i in range(l):
	p_m6.append(float(lines[i]))

p_m7 = []
lines = src_m7.readlines()
for i in range(l):
	p_m7.append(float(lines[i]))

p_m4b = []
lines = src_m4_b.readlines()
for i in range(l):
	p_m4b.append(float(lines[i]))

p_m5b = []
lines = src_m5_b.readlines()
for i in range(l):
	p_m5b.append(float(lines[i]))

p_m6b = []
lines = src_m6_b.readlines()
for i in range(l):
	p_m6b.append(float(lines[i]))

#font = {'fontname':'Times'}
plt.figure(figsize=(10,8))
plt.rc('font', family='Times New Roman')

plt.subplot(121)
plt.title('Random Query List')
plt.plot(x, p_m1, label='1 Stream(LS)')
plt.plot(x, p_m2, label='1S+Concat FC(LS)')
plt.plot(x, p_m3, label='2S w/o REP(LS)')
plt.plot(x, p_m7, label='2S+CCL(LS)')
plt.plot(x, p_m4, marker = '^', linewidth=1, label='RepNet+CRL(LS)')
plt.plot(x, p_m5, marker = '^', linewidth=1, label='RepNet+SRL(LS)')
plt.plot(x, p_m6, marker = '^', linewidth=1, label='RepNet+PRL(LS)')
plt.plot(x, p_m4b, marker = '.', label='RepNet+CRL(BS)')
plt.plot(x, p_m5b, marker = '.', label='RepNet+SRL(BS)')
plt.plot(x, p_m6b, marker = '.', label='RepNet+PRL(BS)')
plt.ylim(0.2,0.9)
plt.xlim(2,40)
plt.xlabel('Top_k')
plt.ylabel('Precision@k')
plt.grid(True)
plt.legend()

src_m1 = open('../exp_res/res/pre-v4_2048_3_precision_at_k.txt')
src_m2 = open('../exp_res/res/pre-v7_6144_3_precision_at_k.txt')
src_m3 = open('../exp_res/res/V2_cmp_2348_3_precision_at_k.txt')
src_m4 = open('../exp_res/res/v2_2348_3_precision_at_k.txt')
src_m5 = open('../exp_res/res/V5_2348_3_precision_at_k.txt')
src_m6 = open('../exp_res/res/v6_2348_3_precision_at_k.txt')
src_m7 = open('../exp_res/res/cvpr_n_2048_3_precision_at_k.txt')
src_m4_b = open('../exp_res/res/v2_bucket2-2_3_precision_at_k.txt')
src_m5_b = open('../exp_res/res/v5_bucket2-2_3_precision_at_k.txt')
src_m6_b = open('../exp_res/res/v6_bucket2-2_3_precision_at_k.txt')
l = 100
x = [i for i in range(1,1+l)]
p_m1 = []
lines = src_m1.readlines()
for i in range(l):
	p_m1.append(float(lines[i]))

p_m2 = []
lines = src_m2.readlines()
for i in range(l):
	p_m2.append(float(lines[i]))

p_m3 = []
lines = src_m3.readlines()
for i in range(l):
	p_m3.append(float(lines[i]))

p_m4 = []
lines = src_m4.readlines()
for i in range(l):
	p_m4.append(float(lines[i]))

p_m5 = []
lines = src_m5.readlines()
for i in range(l):
	p_m5.append(float(lines[i]))

p_m6 = []
lines = src_m6.readlines()
for i in range(l):
	p_m6.append(float(lines[i]))

p_m7 = []
lines = src_m7.readlines()
for i in range(l):
	p_m7.append(float(lines[i]))

p_m4b = []
lines = src_m4_b.readlines()
for i in range(l):
	p_m4b.append(float(lines[i]))

p_m5b = []
lines = src_m5_b.readlines()
for i in range(l):
	p_m5b.append(float(lines[i]))

p_m6b = []
lines = src_m6_b.readlines()
for i in range(l):
	p_m6b.append(float(lines[i]))

plt.subplot(122)
plt.title('Tough Query List')
plt.plot(x, p_m1, label='1 Stream(LS)')
plt.plot(x, p_m2, label='1S+Concat FC(LS)')
plt.plot(x, p_m3, label='2S w/o REP(LS)')
plt.plot(x, p_m7, label='2S+CCL(LS)')
plt.plot(x, p_m4, marker = '^', linewidth=1, label='RepNet+CRL(LS)')
plt.plot(x, p_m5, marker = '^', linewidth=1, label='RepNet+SRL(LS)')
plt.plot(x, p_m6, marker = '^', linewidth=1, label='RepNet+PRL(LS)')
plt.plot(x, p_m4b, marker = '.', label='RepNet+CRL(BS)')
plt.plot(x, p_m5b, marker = '.', label='RepNet+SRL(BS)')
plt.plot(x, p_m6b, marker = '.', label='RepNet+PRL(BS)')
plt.ylim(0.2,0.9)
plt.xlim(2,40)
plt.xlabel('Top_k')
plt.ylabel('Precision@k')
plt.grid(True)
plt.legend()

plt.show()
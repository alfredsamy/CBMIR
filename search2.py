# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
from scipy import misc
import numpy as np
import random
import sys
import heapq as hq
import shutil
import os
import gist #https://github.com/yuichiroTCY/lear-gist-python


# gist descriptor of size 960
def gist_descriptor(img):
	return []
	# return gist.extract(img)

def load_data(name):
	with open(name, 'rb') as f:
		save = pickle.load(f)
		desc = save['desc7']
		labels = save['train_labels']
		names = save['name']
		print(len(desc))
		print(len(labels))
		del save
		print('loaded')

	return desc, labels, names

def load_result():
	with open('retrieval_result', 'rb') as f:
		save = pickle.load(f)
		ResultQ = save['query']
		ResultR = save['result']
		indexQ = save['query_index']
		indexR = save['result_index'] 
	print(len(ResultQ))
	print(len(ResultR))
	del ResultR[0]
	del indexR[0]
	return ResultQ, ResultR, indexQ, indexR
	
# calculate ecludian distance
def distance(a,b):
	dsum = 0.
	for i in range(len(a)):
		dsum += (a[i]-b[i])**2
	return dsum

# calculate RBC code for image (image.x == image.y )
def RBC(image, steps = 16):
	#	discrete_radon_transform
	R = np.zeros((steps, len(image)), dtype='float64')
	for s in range(steps):
		rotation = misc.imrotate(image, -s*180/steps).astype('float64')
		R[s] = sum(rotation)
	R = R.reshape(-1)
	
	#RBC
	mx = max(R)
	for j in range(len(R)):
		R[j] /= mx
	median = np.median(R)
	for j in range(len(R)):
		if(R[j] < median):
			R[j] = 0
		else:
			R[j] = 1
	return R


if __name__ == "__main__":
	desc, labels, names = load_data('train_desc')
	t_desc, t_labels,t_names = load_data('test_desc')
	ResultQ, ResultR, indexQ, indexR = load_result()

	# Make result directory
	dst_root = 'Retrieval22'
	shutil.rmtree(dst_root, True)
	os.makedirs(dst_root)
	src_root = '/media/alfred/B8308B68308B2D06/IRMA/ImageCLEFmed2009_train.02/'
	src_root2 = '/media/alfred/B8308B68308B2D06/IRMA/ImageCLEFmed2009_test.03/'

	test_num = -1
	resize = 192
	eval_res = []
	eval_res2 = []
	top_n = 5
	# for t in range(len(t_desc)):
	for t in range(len(ResultQ)):

		gist_test_img = misc.imread(ResultQ[t],mode='RGB')
		gist_test_img = gist_descriptor(gist_test_img)
		

		rbc_test_img = misc.imread(ResultQ[t],mode='L')
		rbc_test_img = misc.imresize(rbc_test_img,(resize,resize))
		rbc_test_img = RBC(rbc_test_img)
		fc7_test_img = t_desc[t]

		test_num += 1
		fc7_retrievals = []
		rbc_retrievals = []
		gist_retrievals = []
		combined_ret = []
		retrievals = []

		combined_ret2 = []
		retrievals2 = []
	
		for i in range(len(ResultR[t])):
			# Calculate different distances
			gist_img = gist_descriptor(misc.imread(ResultR[t][i],mode='RGB'))
			
			rbc_img = misc.imread(ResultR[t][i],mode='L')
			rbc_img = misc.imresize(rbc_img,(resize,resize))
			rbc_img = RBC(rbc_img)

			rbc_dist = distance(rbc_img, rbc_test_img)
			fc7_dist = distance(desc[indexR[t][i]], fc7_test_img)
			gist_dist = distance(gist_test_img, gist_img)
			
			rbc_retrievals.append(rbc_dist)
			fc7_retrievals.append(fc7_dist)
			gist_retrievals.append(gist_dist)

		# Normalize the 2 result and combine them
		fc7_min = min(fc7_retrievals)
		rbc_min = min(rbc_retrievals)
		
		fc7_max = max(fc7_retrievals)-fc7_min
		rbc_max = max(rbc_retrievals)-rbc_min
		gist_max = max(gist_retrievals)
		for i in range(len(fc7_retrievals)):
			fc7_retrievals[i] = (fc7_retrievals[i] - fc7_min ) / fc7_max
			rbc_retrievals[i] = (rbc_retrievals[i] - rbc_min ) / rbc_max
			# gist_retrievals[i] /= gist_max
			hq.heappush(combined_ret, (0.7*fc7_retrievals[i]+0.3*rbc_retrievals[i], i))
			# hq.heappush(combined_ret2, (0.6*fc7_retrievals[i]+0.2*rbc_retrievals[i]+0.2*gist_retrievals[i], i))
			# hq.heappush(combined_ret, (fc7_retrievals[i], i))
		
		# retrieve top N result
		for x in range(top_n):
			retrievals.append(hq.heappop(combined_ret))
			# retrievals2.append(hq.heappop(combined_ret2))
		correct = sum([1 for j in [labels[ indexR[t][u[1]] ] for u in retrievals] if j == t_labels[t]])
		eval_res += [correct / top_n]

		# correct2 = sum([1 for j in [labels[ indexR[t][u[1]] ] for u in retrievals2] if j == t_labels[t]])
		# eval_res2 += [correct2 / top_n]
		
		# print('[query label] ', t_names[t],' ',t_labels[t])
		# print('[names]', [names[u[1]] for u in retrievals])
		# print('[labels]', correct, [labels[i[1]] for i in retrievals])
		# print()
		print(t, "==>", correct / top_n, "==>", sum(eval_res) / len(eval_res))
				 # "||", correct2 / top_n, "==>", sum(eval_res2) / len(eval_res2))

		
		# Save output images
		dst_cur = dst_root + '/test_' + str(test_num) + '_c' + str(correct)
		os.makedirs(dst_cur)

		src = ResultQ[t]
		dst = dst_cur + '/query_' + t_labels[t]
		shutil.copyfile(src, dst)

		for j in range(len(retrievals)):
			src = ResultR[t][retrievals[j][1]]

			dst = dst_cur + '/' + str(j) + '_' + labels[ indexR[t][retrievals[j][1]] ] + '.png'
			shutil.copyfile(src, dst)
	
	print('[Mean Accuracy]', sum(eval_res) / len(eval_res))
	print('[Mean Accuracy]', sum(eval_res2) / len(eval_res2))
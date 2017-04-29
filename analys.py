# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import numpy as np
#from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

if __name__ == '__main__':
	# file = 'train_codes_02.csv'
	file = 'test_codes_03.csv'
	fo = open(file, 'r')
	labels = [[]] #image_id;irma_code;05_class;05_irma_code_submask;06_class;06_irma_code;07_irma_code;08_irma_code
	for str in fo.read().split('\n'):
		labels.append(str.split(';'))
	del labels[1]
	del labels[0]
	fo.close()

	# dic = {}
	# for i in labels:
	# 	x=2
	# 	if(i[x] in dic):
	# 		dic[i[x]] += 1
	# 	else:
	# 		dic[i[x]] = 1
	# for k in dic:
	# 	print(dic[k],"\t",k)
	# print(len(dic))
	d = {}
	for i in labels:
		d[i[0]] = i[2]

	# Train dataset
	# f = open("dataset_labels", 'wb')
	# save = {
	# 	'labels': d,
	# }
	# pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	# f.close()

	# Test dataset
	f = open("test_labels", 'wb')
	save = {
		'labels': d,
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()
import random, os, sys
import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 
# import keras.backend.tensorflow_backend as tfback


# print("tf.__version__ is", tf.__version__)
# print("tf.keras.__version__ is:", tf.keras.__version__)

# def _get_available_gpus():
#     """Get a list of available gpu devices (formatted as strings).

#     # Returns
#         A list of available GPU devices.
#     """
#     #global _LOCAL_DEVICES
#     if tfback._LOCAL_DEVICES is None:
#         devices = tf.config.list_logical_devices()
#         tfback._LOCAL_DEVICES = [x.name for x in devices]
#     return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

# tfback._get_available_gpus = _get_available_gpus

import keras
from tqdm import tqdm
import config
from keras.utils.multi_gpu_utils import multi_gpu_model
import model
from read_recommend_data import split_session,get_dict_items
import argparse

from tensorflow.python.eager import context
G = context.num_gpus()

parser = argparse.ArgumentParser()
parser.add_argument("--pretrain")
#parser.add_argument("--ngpus")
args = parser.parse_args()






def toOneHot(pos,no_items):
	y = np.zeros(no_items)
	y[pos]=1
	return y



'''
	---------------------MODEL PARAMETERS-----------------------
'''
d_emb = config.d_emb
max_len = config.max_len
no_block = config.no_block
n_head = config.n_head
d_model = config.d_model
dropout = config.dropout
dff = config.dff
batch_size = config.batch_size
epochs = config.epochs
session_train,label_train = split_session('recommand_data.json',max_len)
dict_item = get_dict_items('recommand_data.json')
no_items = len(dict_item.keys())


'''
	-----------------BUILD MODEL--------------------
'''
# OLDER VERSION (Multi GPUs)
# G = int(args.ngpus)
# if G <= 1:
# 	print("[INFO] training with 1 GPU...")
# 	model = model.make_model()
# 	if args.pretrain != "":
# 		model.load_weights(args.pretrain)
# 		print("Load pretrain successful!")
# else:
# 	print("[INFO] training with {} GPUs...".format(G))
# 	with tf.device("/cpu:0"):
# 		model = model.make_model()
# 		if args.pretrain != "":
# 			model.load_weights(args.pretrain)
# 			print("Load pretrain successful!")
# 	model = multi_gpu_model(model, gpus=G)
	
#NEW VERSION (Multi GPUs)

model = model.make_model()

if args.pretrain != "":
	model.load_weights(args.pretrain)
	print("Load pretrain successful!")
model = multi_gpu_model(model)

'''
	TRAINING MODEL
'''

train_session = np.vectorize(dict_item.get)(session_train)
label_train = np.vectorize(dict_item.get)(label_train)

print("number of items = {}".format(no_items))
# model.fit(x_test,y=y_test,epochs=epochs,validation_data=[x_validate,y_validate],callbacks=[mrr],batch_size=batch_size)
for epoch in tqdm(range(epochs)):
	#print("Epoch : {}".format(epoch))
	for i in range(train_session.shape[0]//(batch_size*G)):
		train_x = train_session[i*batch_size*G:(i+1)*batch_size*G]
		temp = label_train[i*batch_size*G:(i+1)*batch_size*G]
		train_y = []
		for j in temp:
			train_y.append(toOneHot(j,no_items))
		train_y = np.asarray(train_y)
		model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
		model.fit(train_x,y=train_y,batch_size=batch_size,verbose=0)
	if epoch % 50 == 0:
		model.save_weights('attention_model_{}.h5'.format(str(epoch)))
#mrr.get_data()


#Save model

model.save_weights('attention_model.h5')

import model
import numpy as np
import config
from read_recommend_data import get_dict_items,split_session
import keras.backend as K
from keras.layers import Lambda
import tensorflow as tf

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

model = model.make_model()

model.load_weights('attention_model.h5')
#testing



train_session = np.vectorize(dict_item.get)(session_train)
label_train = np.vectorize(dict_item.get)(label_train)



dct_reverse_item = {}
for key in dict_item.keys():
    dct_reverse_item[dict_item[key]]=key
#print(dict_item)

from time import time
t = time()
x_test = np.random.randint(low=1,high=no_items,size=(batch_size,max_len))
res = model.predict(x_test,batch_size=batch_size)
print("Predict total time {} seconds for {} sessions".format(time()-t,batch_size))
# for i in np.random.randint(low=0,high=batch_size,size=5):
#     print("Session {} : {}".format(i,x_test[i]))
#     print("Predict : {}".format(K.eval(K.argmax(res[i]))))
#     print("Real predict item id : {}".format(dct_reverse_item[K.eval(K.argmax(res[i]))]))
#     print("Real session id : ")
#     for j in x_test[i]:
#         print(dct_reverse_item[j],end=",")
#     print("")

res = model.predict(train_session[:batch_size],batch_size=batch_size)
for i in np.random.randint(low=0,high=batch_size,size=5):
    print("SESSION : ")
    print(train_session[i])
    print("Predict : {}".format(res[i].argsort()[-4:][::-1]))
    print("Real predict item id :")
    for j in res[i].argsort()[-4:][::-1]:
        print(dct_reverse_item[j],end=",")
    print("")
    print("Real session id : ")
    for j in train_session[i]:
        print(dct_reverse_item[j],end=",")
    print("")
    print("Real label : {}".format(dct_reverse_item[label_train[i]]))
    print("")
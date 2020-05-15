from read_recommend_data import get_dict_items,split_session
import config 

import random, os, sys
import numpy as np
import pandas as pd
import keras
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from tqdm import tqdm
# import tensorflow as tf
import keras.backend as K


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

class NormalizeLayer(Layer):
    def __init__(self,epsilon=1e-6,**kwargs):
        super(NormalizeLayer,self).__init__(**kwargs)
        self.eps = epsilon
    
    def build(self,input_shape):
        self.gamma = self.add_weight(name='gamma',shape = input_shape[-1:],initializer=Ones(),trainable=True)
        self.beta = self.add_weight(name='beta',shape = input_shape[-1:],initializer=Zeros(),trainable=True)
        super(NormalizeLayer,self).build(input_shape)
        
    def call(self,x):
        mean = K.mean(x,axis=-1,keepdims=True)
        std = K.std(x,axis=-1,keepdims=True)
        return self.gamma * (x-mean)/(std+self.eps)+self.beta
    
    def compute_output_shape(self,input_shape):
        return input_shape

class FeedForwardNetwork(Layer):
    def __init__(self,dff,d_model,**kwargs):
        super(FeedForwardNetwork,self).__init__(**kwargs)
        self.norm = NormalizeLayer()
        self.dff = dff
        self.d_model = d_model
    
    def build(self,input_shape):
        self.w1 = self.add_weight(name='w1',shape=(input_shape[-1],self.dff),initializer='uniform',trainable=True)
        self.b1 = self.add_weight(name='b1',shape=(self.dff,),initializer='uniform',trainable=True)
        self.w2 = self.add_weight(name='w2',shape=(self.dff,self.d_model),initializer='uniform',trainable=True)
        self.b2 = self.add_weight(name='b2',shape=(self.d_model,),initializer='uniform',trainable=True)
        super(FeedForwardNetwork,self).build(input_shape)   

    def call(self,x):
        output = K.dot(x,self.w1)+self.b1
        output = K.dot(output,self.w2)+self.b2
        output = self.norm(Add()([output,x]))
        return output

def mask(inputs, queries=None, keys=None, type=None,batch_size=16):
    '''
    Masks paddings on keys or queries to inputs
        inputs: 3d tensor. (N, T_q, T_k)
        queries: 3d tensor. (N, T_q, d)
        keys: 3d tensor. (N, T_k, d)
        e.g.,
        >> queries = K.constant([[[1.],
                            [2.],
                            [0.]]], K.float32) # (1, 3, 1)
        >> keys = K.constant([[[4.],
                        [0.]]], K.float32)  # (1, 2, 1)
        >> inputs = K.constant([[[4., 0.],
                                [8., 0.],
                                [0., 0.]]], K.float32)
        >> mask(inputs, queries, keys, "key")
        array([[[ 4.0000000e+00, -4.2949673e+09],
            [ 8.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
        >> inputs = K.constant([[[1., 0.],
                                [1., 0.],
                                [1., 0.]]], K.float32)
        >> mask(inputs, queries, keys, "query")
        array([[[1., 0.],
            [1., 0.],
            [0., 0.]]], dtype=float32)
    '''

    padding_num = -2 ** 32 + 1
    if type in ("k","key","keys"):
        # generate masks
        masks = K.sign(K.sum(K.abs(keys),axis=-1)) # (N,T_k)
        masks = K.expand_dims(masks,axis=1) # (N,1,T_k)
        masks = K.tile(masks,[1,K.shape(queries)[1],1])

        # apply mask to inputs
        paddings = K.ones_like(inputs) * padding_num
        outputs = K.switch(K.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)

    elif type in ("q", "query", "queries"):
        # Generate masks
        masks = K.sign(K.sum(K.abs(queries),axis=-1))  # (N, T_q)
        masks = K.expand_dims(masks,axis=-1)  # (N, T_q, 1)

        # Apply masks to inputs
        outputs = inputs*masks
    elif type in ("f", "future", "right"):
        len_s = K.shape(inputs)[1]
        bs = K.shape(inputs)[:1]
        batch_eye = K.eye(max_len)
        batch_eye = K.expand_dims(batch_eye,0)
        batch_eye = K.tile(batch_eye,[batch_size,1,1])
        masks = K.cumsum(batch_eye, 1)
        paddings = K.ones_like(masks) * padding_num
        outputs = K.switch(K.equal(masks, 0), paddings, inputs)
    else:
        print("Check if you entered type correctly!")

    return outputs

class ScaledDotProductAttention(Layer):
    '''
    q: Packed queries. 3d tensor. [..., T_q, depth].
    k: Packed keys. 3d tensor. [..., T_k, depth].
    v: Packed values. 3d tensor. [..., T_k, d_v].
    causality: If True, applies masking for future blinding.
    training: boolean for controlling dropout.
    dropout: A floating point number of [0, 1].

    Returns:
        outputs 
    '''
    def __init__(self,dropout_rate=0.1,causality=True,training=True,batch_size=16,**kwargs):
        super(ScaledDotProductAttention,self).__init__(**kwargs)
        self.dropout = Dropout(dropout_rate)
        self.causality = causality
        self.training = training
        self.batch_size = batch_size

    def build(self,input_shape):
        super(ScaledDotProductAttention, self).build(input_shape)
    
    def call(self,x):
        assert isinstance(x,list)
        q = x[0]
        k = x[1]
        v = x[2]
        dk_norm = K.sqrt(K.cast(K.shape(k)[-1],dtype='float32'))
        attn = K.batch_dot(q,k,axes=[2,2])/dk_norm
        attn = mask(attn,q,k,type="key",batch_size=self.batch_size)
        if self.causality:
            attn = mask(attn,type="future",batch_size=self.batch_size)
        attn = K.softmax(attn)
        attn = self.dropout(attn,training=self.training)
        output = K.batch_dot(attn,v)
        return output

class MultiHeadAttention(Layer):
    def __init__(self,n_head=8,d_model=512,dropout=0.1,training=True,causality=True,batch_size=16,**kwargs):
        super(MultiHeadAttention,self).__init__(**kwargs)
        self.dropout = dropout
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model//n_head
        self.list_q = []
        self.list_k = []
        self.list_v = []
        self.batch_size = batch_size
        self.causality = causality
        self.training = training
        self.attention = ScaledDotProductAttention(causality=self.causality,training=self.training,batch_size=self.batch_size)
        self.d_model = d_model

    def build(self,input_shape):
        for i in range(self.n_head):
            weights_q = self.add_weight(name='dense_q{}'.format(i),shape=(input_shape[0][-1],self.d_k),initializer='uniform',trainable=True)
            weights_k = self.add_weight(name='dense_k{}'.format(i),shape=(input_shape[1][-1],self.d_k),initializer='uniform',trainable=True)
            weights_v = self.add_weight(name='dense_v{}'.format(i),shape=(input_shape[2][-1],self.d_v),initializer='uniform',trainable=True)
            bias_q = self.add_weight(name='bias_q{}'.format(i),shape=(self.d_k,),initializer='uniform',trainable=True)
            bias_k = self.add_weight(name='bias_k{}'.format(i),shape=(self.d_k,),initializer='uniform',trainable=True)
            bias_v = self.add_weight(name='bias_v{}'.format(i),shape=(self.d_v,),initializer='uniform',trainable=True)
            self.list_q.append([weights_q,bias_q])
            self.list_k.append([weights_k,bias_k])
            self.list_v.append([weights_v,bias_v])
        self.w_o = self.add_weight(name='w_o',shape=(self.n_head*self.d_k,self.d_model),initializer='uniform',trainable=True)
        self.b_o = self.add_weight(name='b_o',shape=(self.d_model,),initializer='uniform',trainable=True)
        super(MultiHeadAttention,self).build(input_shape)

    def call(self,x):
        assert isinstance(x,list)
        q = x[0]
        k = x[1]
        v = x[2]
        heads = []
        for i in range(n_head):
            qs = K.dot(q,self.list_q[i][0])+self.list_q[i][1]
            ks = K.dot(k,self.list_k[i][0])+self.list_k[i][1]
            vs = K.dot(v,self.list_v[i][0])+self.list_v[i][1]
            head = self.attention([qs,ks,vs])
            heads.append(head)
        head = Concatenate()([x for x in heads])
        output = K.dot(head,self.w_o)+self.b_o
        output = Dropout(self.dropout)(output,training=self.training)
        return output


def getPosEncodingMatrix(max_len,d_emb):
    '''
    Position Encoding in transformers:
    Formula : 
        PE(pos,2i) = sin(pos/(10000^(2i/d_model)))
        PE(pos,2i+1) = cos(pos/(10000^(2i/d_model)))

    Returns:
        All the position encoding matrix value (maxlen,d_model)
    '''
    PE_value = []
    for pos in range(max_len):
        t = []
        for i in range(d_emb):
            if i%2==0:
                pe = np.sin(pos/(10000)**(i/d_emb))
            else:
                pe = np.cos(pos/(10000)**((i-1)/d_emb))
            t.append(pe)
        PE_value.append(t)
    return PE_value

def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.
    Returns
    weight variable: (V, E)
    '''
    embedding_layer = Embedding(input_dim=vocab_size,output_dim=num_units,mask_zero=zero_pad)
    return embedding_layer



class model_layer(Layer):
    def __init__(self,no_block,block_type="global",batch_size=16,d_model=16,**kwargs):
        super(model_layer,self).__init__(**kwargs)
        self.no_block = no_block
        self.block_type = block_type
        self.mha1 = MultiHeadAttention(n_head,d_model,dropout,batch_size=batch_size)
        self.mha2 = MultiHeadAttention(n_head,d_model,dropout,causality=False,batch_size=batch_size)
        self.ffw = FeedForwardNetwork(dff,d_model)
        self.output_dim = d_model
    def build(self,input_shape):
        super(model_layer,self).build(input_shape)

    def call(self,x):
        output = x
        for i in range(self.no_block):
            output = self.mha1([output,output,output])
            output = self.mha2([output,output,output])
            output = self.ffw(output)
        if self.block_type in ("gl","global","globals"):
            return output[:, -1, :]
        else:
            return K.sum(output,axis=-2)

    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.output_dim)



class Result(Layer):
    def __init__(self,d_emb, emb,no_items=7,**kwargs):
        self.d_emb = d_emb
        self.emb = emb
        self.no_items = no_items
        super(Result,self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.w = self.add_weight(name='w',shape=(self.d_emb,self.d_emb*2),initializer='uniform',trainable=True)
        self.b = self.add_weight(name='b',shape=(self.d_emb*2,),initializer='uniform',trainable=True)
        super(Result,self).build(input_shape)

    def call(self,x):
        res = x
        emb_temp = self.emb
        emb_temp = K.dot(emb_temp,self.w)+self.b
        res = K.dot(res,K.transpose(emb_temp))
        return K.softmax(res)
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0],self.no_items)


def make_model():
    pe_emb = getPosEncodingMatrix(max_len,d_emb)
    pe_emb = np.asarray(pe_emb)
    pe_emb[0,:] = 0
    # print(pe_emb)

    embedding_layer = get_token_embeddings(no_items,d_emb)
    x = Input(shape=(max_len,))
    #x = K.constant(t)
    embed = embedding_layer(x)
    embedding_final = Lambda(lambda x : x+pe_emb)(embed)
    global_embedding = model_layer(no_block,block_type="global",batch_size=batch_size,d_model=d_model)(embedding_final)
    local_embedding = model_layer(no_block,block_type="local",batch_size=batch_size,d_model=d_model)(embedding_final)
    final_att = Concatenate(axis=1)([global_embedding,local_embedding])
    result_layer = Result(d_emb, embedding_layer.embeddings,no_items=no_items)
    y = result_layer(final_att)
    model = Model(x,y)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model
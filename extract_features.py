
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os,sys
import argparse

import datetime
import pandas as pd
from model import ResNetModel
from tqdm import tqdm
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor


# In[6]:


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir", dest="model", type=str, default="../training/resnet_20181022_150646/checkpoint/", help="Resnet101 Net Model File")
parser.add_argument("--num_classes",dest="num_classes",type=int,default=5)
parser.add_argument("--val_file",dest="val_file",type=str,default="../data/val.txt")
parser.add_argument("--batch_size",dest="batch_size",type=int,default=32)
parser.add_argument("--features",dest="features",type=str,default=None)
parser.add_argument("--label",dest="label",type=str,default=None)
args = parser.parse_args()

#checkpoint_dir="../training/resnet_20181022_150646/checkpoint/"
checkpoint_path=tf.train.latest_checkpoint(args.model)


#num_classes=5
#val_file="../data/val.txt"


# In[7]:


val_preprocessor = BatchPreprocessor(dataset_file_path=args.val_file, num_classes=args.num_classes, output_size=[224, 224])
#batch_size=32
batch_size=args.batch_size
val_num_batches = np.floor(len(val_preprocessor.labels) / batch_size).astype(np.int16)


# In[4]:


# Placeholders
x = tf.placeholder(tf.float32, [None,224, 224, 3])
y = tf.placeholder(tf.float32, [None,args.num_classes])
is_training = tf.placeholder('bool', [])


# Model
model = ResNetModel(is_training, depth=101, num_classes=args.num_classes)
model.inference(x)


# In[9]:


saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,checkpoint_path)
    
    avg_pool=sess.graph.get_tensor_by_name("avg_pool:0")
    
    label_onehot=np.zeros((val_num_batches,args.batch_size,args.num_classes))
    features=np.zeros((val_num_batches,args.batch_size,2048))
    #features=[]
    for batch in tqdm(range(val_num_batches)):
        batch_x,batch_y=val_preprocessor.next_batch(batch_size=args.batch_size)
        AvgPool=sess.run(avg_pool,feed_dict={x:batch_x,is_training:False})
        
        #features.append(AvgPool)
        features[batch,:,:]=AvgPool
        label_onehot[batch,:,:]=batch_y
       
#np.save("./val-label.npy",label_onehot)
#np.save("./val-features.npy",features)
np.save(args.label,label_onehot)
np.save(args.features,features)

#x=np.stack(feartures,axis=0)
#x=np.reshape(x,[-1,2048])

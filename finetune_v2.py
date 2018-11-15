
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd


import random
import os,sys
import datetime

from model import ResNetModel
sys.path.insert(0, '../utils')


# In[2]:


class ImageData:

    def __init__(self, batch_size, load_size, channels, augment_flag,num_class):
        self.batch_size = batch_size
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag
        self.num_class=num_class
        self.mean_color=tf.constant([132.2766, 139.6506, 146.9702],name="mean_color")

    def image_processing(self, filename,label):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)
                
        lab=tf.cast(label,tf.int32)
        lab=tf.one_hot(lab,self.num_class,dtype=tf.float32)
        img=tf.subtract(img,self.mean_color)
        return img,lab


def augmentation(image, augment_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image


# In[ ]:


tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 101, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 26, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc', 'Finetuning layers, seperated by commas')

tf.app.flags.DEFINE_string('training_file', '../data/train_origin.txt', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/val.txt', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')

FLAGS = tf.app.flags.FLAGS


img_size=224
img_ch=3
augment_flag=True


# In[4]:


#val_file="../data/val.txt"
#training_file="../data/train_origin.txt"
train_data=pd.read_csv(FLAGS.training_file,header=None,sep=" ")
train_images=[str(s) for s in train_data.iloc[:,0]]
train_labels=[int(l) for l in train_data.iloc[:,1]]

val_data=pd.read_csv(FLAGS.val_file,header=None,sep=" ")
val_images=[str(s) for s in val_data.iloc[:,0]]
val_labels=[int(l) for l in val_data.iloc[:,1]]



#mean_color=[132.2766, 139.6506, 146.9702]

image=tf.placeholder(tf.string,shape=[None])
label=tf.placeholder(tf.int32,shape=[None])
batch_size=tf.placeholder(tf.int64)

x = tf.placeholder(tf.float32, [None, img_size,img_size,img_ch])
y = tf.placeholder(tf.float32, [None,FLAGS.num_classes])
is_training = tf.placeholder('bool', [])
tra_num_batches=len(train_images)//FLAGS.batch_size
val_num_batches=len(val_images)//FLAGS.batch_size
#mean_color=tf.constant(mean_color)


# In[ ]:


def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('resnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.tensorboard_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('resnet_depth={}\n'.format(FLAGS.resnet_depth))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))

    flags_file.write('tensorboard_root_dir={}\n'.format(FLAGS.tensorboard_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()
    
    tra_Image_Data_Class = ImageData(FLAGS.batch_size, img_size,img_ch,True,FLAGS.num_classes)
    train_dataset=tf.data.Dataset.from_tensor_slices((train_images,train_labels))
    train_dataset = train_dataset.map(tra_Image_Data_Class.image_processing, num_parallel_calls=8).shuffle(10000).prefetch(FLAGS.batch_size).batch(FLAGS.batch_size).repeat()

    train_iterator=train_dataset.make_initializable_iterator()
    tra_img,tra_lab=train_iterator.get_next()


    val_Image_Data_Class = ImageData(FLAGS.batch_size, img_size,img_ch,False,FLAGS.num_classes)
    val_dataset=tf.data.Dataset.from_tensor_slices((val_images,val_labels))
    val_dataset = val_dataset.map(val_Image_Data_Class.image_processing, num_parallel_calls=8).shuffle(10000).prefetch(FLAGS.batch_size).batch(FLAGS.batch_size).repeat()

    val_iterator=val_dataset.make_initializable_iterator()
    val_img,val_lab=val_iterator.get_next()
    
    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = ResNetModel(is_training, depth=FLAGS.resnet_depth, num_classes=FLAGS.num_classes)
    loss = model.loss(x, y)
    train_op = model.optimize(FLAGS.learning_rate, train_layers)

    # Training accuracy of the model
    correct_pred = tf.equal(tf.argmax(model.prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Summaries
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        sess.run(train_iterator.initializer,feed_dict={image:train_images,label:train_labels,batch_size:FLAGS.batch_size})
        # Load the pretrained weights
        model.load_original_weights(sess, skip_layers=train_layers)

        print("{} Start training...".format(datetime.datetime.now()))
        for epoch in range(FLAGS.num_epochs):
            step=1
            print("{} Epoch number: {}".format(datetime.datetime.now(), epoch+1))
            for idx in range(tra_num_batches):
                batch_x,batch_y=sess.run([tra_img,tra_lab])

                _,tra_loss,tra_acc=sess.run([train_op,loss,accuracy],feed_dict={x:batch_x,y:batch_y,is_training:True})

                if step%FLAGS.log_step==0:
                   #print("[ Epoch:%d ],Step:%d,** loss:%f,accuracy:%f **"%(epoch,step,tra_loss,tra_acc))
                   s = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y, is_training: False})
                   train_writer.add_summary(s, epoch * tra_num_batches + step)
                step+=1

            print("{} Start validation".format(datetime.datetime.now()))
            test_acc = 0.
            test_count = 0

            sess.run(val_iterator.initializer,feed_dict={image:val_images,label:val_labels,batch_size:FLAGS.batch_size})
            for _ in range(val_num_batches):
                batch_x,batch_y=sess.run([val_img,val_lab])
                val_loss,val_acc=sess.run([loss,accuracy],feed_dict={x:batch_x,y:batch_y,is_training:False})

                test_acc+=val_acc
                test_count+=1

            test_acc/=test_count
            s = tf.Summary(value=[
                    tf.Summary.Value(tag="validation_accuracy", simple_value=test_acc)
                ])
            val_writer.add_summary(s, epoch+1)

            print("{} Validation Accuracy = {:.4f}".format(datetime.datetime.now(), test_acc))

            print("{} Saving checkpoint of model...".format(datetime.datetime.now()))
            #save checkpoint of the model
            checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_path)
            print("{} Model checkpoint saved at {}".format(datetime.datetime.now(), checkpoint_path))


            
if __name__ == '__main__':
    tf.app.run()


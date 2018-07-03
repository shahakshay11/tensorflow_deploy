import tensorflow as tf
import numpy as np
import os,sys


DATA_SIZE = 100
SAVE_PATH = './save'
EPOCHS = 1000
LEARNING_RATE = 0.01
MODEL_NAME = 'test'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)


#Declaring training and test data
data = (np.random.rand(DATA_SIZE,2),np.random.rand(DATA_SIZE,1))
test = (np.random.rand(DATA_SIZE//8,2),np.random.rand(DATA_SIZE //8,1))

#resetting the default graph for fresh computation graph to follow
tf.reset_default_graph()


#declaring tensors to store the data inputs and targets
x = tf.placeholder(tf.float32,shape=[None,2],name='inputs')
y = tf.placeholder(tf.float32,shape=[None,1],name='targets')

#defining a neural net with 16 activation units and 2 layers
net = tf.layers.dense(x,16,activation=tf.nn.relu)
net = tf.layers.dense(net,16,activation=tf.nn.relu)
pred = tf.layers.dense(net,1,activation=tf.nn.sigmoid,name='prediction')

#MSE loss declaration
loss = tf.reduce_mean(tf.squared_difference(y,pred),name='loss')

#declare the ADAM optimizer
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

#declare the checkpoint to be used for loading of model after curr session is executed
checkpoint = tf.train.latest_checkpoint(SAVE_PATH)
should_train = checkpoint == None

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if should_train:
        print("Training")
        #The `Saver` class adds ops to save and restore variables to and from
        # *checkpoints*.  It also provides convenience methods to run these ops
        saver = tf.train.Saver()
        for epoch in range(EPOCHS):
            _,curr_loss = sess.run([train_step,loss],feed_dict=
                {x:data[0],y:data[1]})
            print('EPOCH = {}, LOSS = {:0.4f}'.format(epoch, curr_loss))
        path = saver.save(sess,SAVE_PATH + '/' + MODEL_NAME + '.ckpt')
        print("saved at {}".format(path))
    else:
        print("Restoring")
        graph = tf.get_default_graph()
        saver = tf.train.import_meta_graph(checkpoint + '.meta')
        saver.restore(sess,checkpoint)

        loss = graph.get_tensor_by_name('loss:0')

        test_loss = sess.run(loss,feed_dict = {'inputs:0':test[0],'targets:0':test[1]})

        print(sess.run(pred,feed_dict={'inputs:0':np.random.rand(10,2)}))

        print("TEST LOSS = {:0.4f}".format(test_loss))


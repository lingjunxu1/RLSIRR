from __future__ import division
import os,cv2,scipy.io
import tensorflow as tf
import tf_slim as slim
import numpy as np
class Opts:
    def __init__(self):
        self.gpu_id = "0"
        self.task = "./baseline/Kim/ckpts"
        self.data = "/media/sharesto/data/SM/data/"
        self.save_model_freq = 3
        self.is_hyper = 1
        self.is_training = 0
        self.continue_training = False
opt = Opts()      
os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu_id    
gpu_id = opt.gpu_id
task=opt.task
is_training=opt.is_training
continue_training=opt.continue_training
hyper=opt.is_hyper==1     
EPS = 1e-12
channel = 64 # number of feature channels to build the model, set to 64
train_syn_root=opt.data


def build_net(ntype, nin, nwb=None, name=None):
    if ntype == 'conv':
        return tf.nn.relu(tf.nn.conv2d(nin, nwb[0], strides=[1, 1, 1, 1], padding='SAME', name=name) + nwb[1])
    elif ntype == 'pool':
        return tf.nn.avg_pool2d(nin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    weights = tf.constant(weights)
    bias = vgg_layers[i][0][0][2][0][1]
    bias = tf.constant(np.reshape(bias, (bias.size)))
    return weights, bias
# build VGG19 to load pre-trained parameters
def build_vgg19(input, vgg_path,reuse=False):
    with tf.compat.v1.variable_scope("vgg19"):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        net = {}
        vgg_layers = vgg_path['layers'][0]
        net['input'] = input - np.array([123.6800, 116.7790, 103.9390]).reshape((1, 1, 1, 3))
        net['conv1_1'] = build_net('conv', net['input'], get_weight_bias(vgg_layers, 0), name='vgg_conv1_1')
        net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2), name='vgg_conv1_2')
        net['pool1'] = build_net('pool', net['conv1_2'])
        net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5), name='vgg_conv2_1')
        net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7), name='vgg_conv2_2')
        net['pool2'] = build_net('pool', net['conv2_2'])
        net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10), name='vgg_conv3_1')
        net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12), name='vgg_conv3_2')
        net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14), name='vgg_conv3_3')
        net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16), name='vgg_conv3_4')
        net['pool3'] = build_net('pool', net['conv3_4'])
        net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19), name='vgg_conv4_1')
        net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21), name='vgg_conv4_2')
        net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23), name='vgg_conv4_3')
        net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25), name='vgg_conv4_4')
        net['pool4'] = build_net('pool', net['conv4_4'])
        net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28), name='vgg_conv5_1')
        net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30), name='vgg_conv5_2')
        return net
def lrelu(x):
    return tf.maximum(x * 0.2, x)
def nm(x):
    w0 = tf.Variable(1.0, name='w0')
    w1 = tf.Variable(0.0, name='w1')
    return w0 * x + w1 * slim.batch_norm(x)
# our reflection removal model
def build(input,vgg_path):
    if hyper:
        print("[i] Hypercolumn ON, building hypercolumn features ... ")
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0,vgg_path)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.compat.v1.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    else:
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0,vgg_path)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.image.resize_bilinear(tf.zeros_like(vgg19_f),(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    net=slim.conv2d(input,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,3*2,[1,1],rate=1,activation_fn=None,scope='g_conv_last') # output 6 channels --> 3 for transmission layer and 3 for reflection layer
    return net
def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0] // 2, shape[1] // 2
        for i in range(np.minimum(shape[2], shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)

    return _initializer

def build_reconnet(input,vgg_path): #BTnet, here in this code it is named as reconstruction net
    if hyper:
        print("[i] Reconnet: Hypercolumn ON, building hypercolumn features ... ")
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0,vgg_path)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.compat.v1.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    else:
        vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0,vgg_path)
        for layer_id in range(1,6):
            vgg19_f = vgg19_features['conv%d_2'%layer_id]
            input = tf.concat([tf.image.resize_bilinear(tf.zeros_like(vgg19_f),(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    net=slim.conv2d(input,channel,[1,1],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv0')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,channel,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,channel,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,channel,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,channel,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,channel,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,channel,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    net=slim.conv2d(net,channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_conv_last') # output 6 channels --> 3 for transmission layer and 3 for reflection layer
    return net
def buildModel():
    SPnet_path = task
    vgg_path = scipy.io.loadmat('./baseline/Kim/ckpts/imagenet-vgg-verydeep-19.mat')
    tf.compat.v1.disable_eager_execution()
    input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
    network = build(input,vgg_path)
    transmission_layer, reflection_layerb4 = tf.split(network, num_or_size_splits=2, axis=3) 
    with tf.compat.v1.variable_scope('R_reconnet'): 
        inputgR = reflection_layerb4
        reflection_layer = build_reconnet(inputgR,vgg_path)
        reflection_layer = tf.identity(reflection_layer, name="reflection_layer")
    ckpt = tf.train.get_checkpoint_state(SPnet_path)
    sess=tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(SPnet_path)
    saver_restore = tf.compat.v1.train.Saver([var for var in tf.compat.v1.trainable_variables()])
    saver_restore.restore(sess, ckpt.model_checkpoint_path)
    
    return sess,transmission_layer, reflection_layerb4, reflection_layer,input,vgg_path

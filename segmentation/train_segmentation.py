import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import pandas as pd
import socket
from pyntcloud import PyntCloud
import os
import sys
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))
import provider
import tf_util
import label_util
from model import *
import dataset

# if os.path.exists("/scratch/thesis/HIL"):
#     import ptvsd
#     ptvsd.enable_attach("thesis", address = ('192.33.89.41', 3000))
#     ptvsd.wait_for_attach()

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--log_dir', default='model', help='Log dir')
parser.add_argument('--num_point', type=int, default=8192, help='Point number')
parser.add_argument('--max_epoch', type=int, default=500000) #better just stop after convergence
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay')
FLAGS = parser.parse_args()


NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


NUM_CLASSES = 21

#setup data provider
TRAIN_DATA = dataset.Block(num_classes = NUM_CLASSES, npoints= NUM_POINT, split = "train")
TEST_DATA = dataset.Block(num_classes = NUM_CLASSES, npoints= NUM_POINT, split = "test")

LOG_DIR = os.path.join(BASE_DIR, FLAGS.log_dir)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def rotate_point_cloud(batch_data):
    #Randomly rotate the point clouds around z-axis
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        # just around z-axis
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])                            
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def get_batch(data, idxs, start_idx, end_idx, with_dropout=False):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_sampleweight = np.zeros((bsize, NUM_POINT), dtype=np.float32)
    for i in range(bsize):
        ps,seg,smpw = data[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_label[i,:] = seg
        batch_sampleweight[i,:] = smpw
    
    if(with_dropout == True):
        dropout_ratio = np.random.random()*0.875 
        drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
        batch_data[i,drop_idx,:] = batch_data[i,0,:]
        batch_label[i,drop_idx] = batch_label[i,0]
        batch_sampleweight[i,drop_idx] *= 0

    return batch_data, batch_label, batch_sampleweight

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, sampleweights_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = get_model(NUM_CLASSES, pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl, sampleweights_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'sampleweights_pl' : sampleweights_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        best_acc = -1
        for epoch in range(MAX_EPOCH):

            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            
            # Save the variables to disk.
            if epoch % 5 == 0:
                acc1 = eval_one_epoch(sess, ops, test_writer)
                acc2 = eval_one_epoch(sess, ops, test_writer)
                acc = (acc1 + acc2) /2
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

                if acc > best_acc:
                    # Save best model to disk.
                    best_acc = acc
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                    log_string("Best Model saved in file: %s" % save_path)
      
def train_one_epoch(sess, ops, train_writer):
    is_training = True
    
    log_string('----')
    
    #shuffle data
    idxs = np.arange(len(TRAIN_DATA))
    np.random.shuffle(idxs)

    num_batches = len(TRAIN_DATA) // BATCH_SIZE 
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 10 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        #get batch 
        batch_data, batch_label, batch_smpw = get_batch(TRAIN_DATA, idxs, start_idx, end_idx, with_dropout=True)
        #rotate data
        batch_data[:,:,0:3] = provider.rotate_point_cloud(batch_data[:,:,0:3])
        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['sampleweights_pl']: batch_smpw,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')

    test_idxs = np.arange(0, len(TEST_DATA))
    num_batches = len(TEST_DATA)//BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_smpw = get_batch(TEST_DATA, test_idxs, start_idx, end_idx)

        batch_data[:,:,0:3] = provider.rotate_point_cloud(batch_data[:,:,0:3])

        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['sampleweights_pl']: batch_smpw,
                     ops['is_training_pl']: is_training,}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0)) # do not take unkown into account
        total_correct += correct
        total_seen += np.sum((batch_label>0) & (batch_smpw>0))
        loss_sum += loss_val
        
        for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
                total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))

    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval point accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    caliweights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    log_string('eval point calibrated average acc: %f' % (np.average(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6),weights=caliweights)))

    return total_correct/float(total_seen)

if __name__ == "__main__":
    train()
    LOG_FOUT.close()

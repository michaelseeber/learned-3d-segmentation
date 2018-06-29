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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import label_util
from model import *


if os.path.exists("/scratch/thesis/HIL"):
    import ptvsd
    ptvsd.enable_attach("thesis", address = ('192.33.89.41', 3000))
    ptvsd.wait_for_attach()



parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required = True)
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='model', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=10000, help='Batch Size during training [default: 3000]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
# parser.add_argument('--test_scene', type=int, default=6, help='Which scene to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()



DATA_PATH = FLAGS.data_path
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = os.path.join(BASE_DIR, FLAGS.log_dir)
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
# os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
# os.system('cp train_segmentation.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

NUM_CLASSES = 41

scene_list = []
with open(os.path.join(DATA_PATH, "list.txt"), "r") as fid:
    for line in fid:
        line = line.strip()
        if line:
            scene_list.append(line)


#Load scenes from disk
pointcloud = []
labels = []
for scene in scene_list:
    pointcloud.append(PyntCloud.from_file(os.path.join(DATA_PATH, scene, (scene + "_vh_clean_2.ply"))).points)
    labels.append(PyntCloud.from_file(os.path.join(DATA_PATH,  scene, (scene + "_vh_clean_2.labels.ply"))).points)
pointcloud = pd.concat(pointcloud)
labels = pd.concat(labels)


# Center PointCloud
pointcloud['x'] = (pointcloud['x'] - pointcloud['x'].mean())
pointcloud['y'] = (pointcloud['y'] - pointcloud['y'].mean())
pointcloud['z'] = (pointcloud['z'] - pointcloud['z'].mean())
train_data = pointcloud.drop('alpha', axis=1).values


todelete = []
# Extract Labels from vertex color
labelcolors = labels.drop(['x','y','z','alpha','label'], axis=1).values
train_label = np.zeros(labelcolors.shape[0], dtype=np.int8)
for i in range(labelcolors.shape[0]):
    color = (labelcolors[i, 0], labelcolors[i, 1], labelcolors[i, 2])
    if(label_util.color2label(color) == 40):
        todelete.append(i)
    train_label[i] = label_util.color2label(color)
train_data = np.delete(train_data, todelete, axis=0)
train_label = np.delete(train_label, todelete, axis=0)


#Print statistic about loaded data
unique_items, counts = np.unique(train_label, return_counts=True)
for i in range(unique_items.size):
    print("Label: %3s   |   Class: %15s   |   Count: %6s" % (unique_items[i], label_util.label2class(unique_items[i]) , counts[i] )) 
# fout = open(os.path.join(DATA_PATH, '_start.obj'), 'w')
# for i in range(train_data.shape[0]):
#     color = label_util.label2color(train_label[i])
#     fout.write('v %f %f %f %d %d %d\n' % (train_data[i,0],train_data[i,1], train_data[i,2], color[0], color[1], color[2]))
# fout.close()

#-----SCANNET LABELING -------
# train_label = np.squeeze(labels.points[['label']].values) #scannet ids
# Map scnanet id's into range 0-31 
# covnert back to scannet id's through SCANNET_MAPPING[train_label]
# SCANNET_MAPPING, train_label = np.unique(train_label, return_inverse=True)

print(train_label[0:100])

test_data = train_data
test_label = train_label



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

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred = get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
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
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
      
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                eval_one_epoch(sess, ops, test_writer)
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

    
        
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string('----')
    current_data, current_label, _ = provider.shuffle_data(train_data, train_label) 
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 10 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string('----')
    current_data = test_data
    current_label = np.squeeze(test_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    print(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))

    # unique_predictions, unique_counts = np.unique(np.concat(prediction), return_counts=True)
    # for i in range(unique_predictions.size):
    #     print("Label: %3s   |   Class: %15s   |   Count: %6s" % (unique_predictions[i], label_util.label2class(unique_predictions[i]) , counts[i])) 





if __name__ == "__main__":
    train()
    LOG_FOUT.close()

import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data'))
from model import *
import label_util
import dataset
import tf_util
from pyntcloud import PyntCloud
import pandas as pd
import csv


# if os.path.exists("/scratch/thesis/HIL"):
#     import ptvsd
#     ptvsd.enable_attach("thesis", address = ('192.33.89.41', 3000))
#     ptvsd.wait_for_attach()

NUM_CLASSES = 21
NUM_POINT = 8192
BATCH_SIZE = 70
TEST_WHOLE_SCENE = dataset.WholeScene(num_classes = NUM_CLASSES, split="test")
MODEL_PATH = "/scratch/thesis/segmentation/models_collection/full_nodropout_newrotation/best_model_epoch_2690.ckpt"
VOXELIZE = True

CALI_FULL = np.array([0.36156243, 0.24771595, 0.04908998, 0.02638608, 0.01511378, 0.02427992, 0.02658031, 0.02507698, 0.00374616, 0.00318649, 0.00279083, 0.01415562, 0.0056411,  0.02418471, 0.00660461, 0.00198952, 0.00497217, 0.00343845, 0.02911088, 0.02357486])
CALI_OFFICE= np.array([3.8166097e-01, 2.3369221e-01, 4.1152272e-02, 2.1433709e-02, 2.1445993e-02, 4.2433705e-02, 1.3419092e-02, 4.1446436e-02, 1.0824659e-03, 7.1552588e-04, 8.0008345e-04, 2.5485119e-02, 3.5791290e-03, 2.0537285e-02, 9.8955249e-03, 2.0925151e-04, 2.4702391e-03, 3.1861218e-03, 1.9040102e-02, 2.6779193e-02])
CALI_HOTEL = np.array([0.36070746, 0.22689487, 0.03900941, 0.01969012, 0.01378298, 0.06537995, 0.00782773, 0.04289294, 0.00141417, 0.00111955, 0.00129543, 0.03584787, 0.00059217, 0.01818161, 0.01053767, 0.00055548, 0.00170424, 0.00262316, 0.01722222, 0.03178674])
CALI_LOUNGE = np.array([0.32420272, 0.30770922, 0.05111435, 0.02756071, 0.01150658, 0.00969614, 0.00616465, 0.06723338, 0.0004018,  0.00063942, 0.0006359, 0.00836682, 0.00307603, 0.0127871,  0.01336045, 0., 0.003152, 0.00198015, 0.02975668, 0.03043082])
CALI_APPARTMENTS = np.array([4.6007186e-01, 1.6846141e-01, 1.2115659e-02, 1.4227793e-02, 3.0626480e-02, 3.9693095e-02, 4.0795407e-03, 3.9922696e-02, 6.3146371e-03, 6.0088621e-03, 5.7357321e-03, 2.6273809e-02, 1.4107006e-02, 2.9455611e-02, 6.0980916e-03, 0.0000000e+00, 9.9230008e-03, 3.6997702e-05, 5.4700013e-02, 1.8962411e-02])
caliweights = CALI_FULL


def conv_weight_variable(name, shape, stddev=1.0):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return tf.get_variable(name, shape, dtype=tf.float32,
                           initializer=initializer)


def bias_weight_variable(name, shape, cval=0.0):
    initializer = tf.constant_initializer(cval)
    return tf.get_variable(name, shape, dtype=tf.float32,
                           initializer=initializer)


def conv3d(x, weights, name=None):
    return tf.nn.conv3d(x, weights, name=name,
                        strides=[1, 1, 1, 1, 1], padding="SAME")


def segment():
    with tf.device('/gpu:0'):
        pointclouds_pl, labels_pl, sampleweights_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        batch = tf.Variable(0)
        pred = get_model(NUM_CLASSES, pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl, sampleweights_pl)

    saver = tf.train.Saver() 

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    # segment pointclouds that have groundtruth for reconstruction available
    scene_list = []
    with open(os.path.join('/scratch/thesis/data/scenes/full/', 'full_test.txt'), "r") as fid:
        for line in fid:
            line = line.strip()
            if line:
                scene_list.append(line)
    
    evaluated_scenes = []
    with tf.Session(config=config) as sess:
        # Restore variables from model_path
        saver.restore(sess, MODEL_PATH)
        print("Model restored")

        ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'sampleweights_pl' : sampleweights_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'step': batch}

        # eval_whole_scene_one_epoch(sess, ops)
        for scene_name in scene_list:
            evaluated_scenes.append(eval_scene(scene_name, sess, ops))\

    if(VOXELIZE):
        for idx, scene in enumerate(evaluated_scenes):
            scene_name =  scene_list[idx]
            np.savez_compressed(os.path.join('/scratch/thesis/data/segmented/oldSeg', 'voxelgrid_%s.npz' % scene_name), volume=scene)
            print('Saving %s datacost to disk' % scene_name)



def eval_scene(scene_name, sess, ops):
    is_training = False
    
    print('----')

    scene_id = dataset.scene_name_to_id(scene_name,split="test")
    batch_data, gt_label, batch_smpw = TEST_WHOLE_SCENE[scene_id]

    scene_end = batch_data.shape[0]

    if batch_data.shape[0]<BATCH_SIZE:
        batch_data = np.concatenate((batch_data, np.zeros((BATCH_SIZE-scene_end,NUM_POINT,6))))
        gt_label = np.concatenate((gt_label, np.zeros((BATCH_SIZE-scene_end,NUM_POINT))))
        batch_smpw = np.concatenate((batch_smpw, np.zeros((BATCH_SIZE-scene_end,NUM_POINT))))
    else:
        print("error")

    feed_dict = {ops['pointclouds_pl']: batch_data,
                 ops['is_training_pl']: is_training,
                 ops['labels_pl']: gt_label,
                 ops['sampleweights_pl']: batch_smpw}
    step,  pred_val = sess.run([ops['step'], ops['pred']],
                                feed_dict=feed_dict)
    pred_class = np.argmax(pred_val, 2) # BxN


    #Stats
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    correct = np.sum((pred_class == gt_label) & (gt_label>0) & (batch_smpw>0)) # evaluate only on 20 categories but not unknown
    seen = np.sum((gt_label>0) & (batch_smpw>0))
    tmp,_ = np.histogram(gt_label,range(22))
    for l in range(NUM_CLASSES):
        total_seen_class[l] += np.sum((gt_label==l) & (batch_smpw>0))
        total_correct_class[l] += np.sum((pred_class==l) & (gt_label==l) & (batch_smpw>0))
    caliacc = np.average(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6),weights=caliweights)
    accuracy = (correct / float(seen))
    avg_class_acc = (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6)))
    per_class = []
    for l in range(1,NUM_CLASSES):
	    per_class.append((l,total_correct_class[l]/float(total_seen_class[l])))
    general_stats = {"scene": scene_name , "accuracy": accuracy, "avg_class_accuray": avg_class_acc, "weighted_class_accuracy": caliacc}
    per_class_stats = {x[0]:x[1:] for x in per_class}
    data = {**general_stats, **per_class_stats}
    add_stat(pd.DataFrame(data, index=[0]))

    #Output Prediction Pointcloud & GT
    fout = open(os.path.join(BASE_DIR, 'results', 'predicted_%s.obj' % scene_name), 'w+')
    for b in range(scene_end):
        for i in range(batch_data.shape[1]):
            color = label_util.label2color(pred_class[b][i], converted=True)
            fout.write('v %f %f %f %d %d %d\n' % (batch_data[b,i,0], batch_data[b,i,1], batch_data[b,i,2], color[0], color[1], color[2]))
    fout.close()

    fout = open(os.path.join(BASE_DIR, 'results', 'groundtruth_%s.obj' % scene_name), 'w+')
    for b in range(scene_end):
        for i in range(batch_data.shape[1]):
            color = label_util.label2color(gt_label[b][i], converted=True)
            fout.write('v %f %f %f %d %d %d\n' % (batch_data[b,i,0], batch_data[b,i,1], batch_data[b,i,2], color[0], color[1], color[2]))
    fout.close()

    #voxelization
    if(VOXELIZE):
        pandasdata = pd.DataFrame(batch_data[0:scene_end, :, 0:3].reshape(-1, 3), columns=['x','y','z'])
        cloud = PyntCloud(pandasdata)
        # use resolutionof 5 cm, bounding box not regular
        voxelgrid_id = cloud.add_structure("voxelgrid", sizes=[0.05, 0.05, 0.05], bb_cuboid=False)
        voxelgrid = cloud.structures[voxelgrid_id]
        # fill voxelgrid with probabilities of points in it ->numerical issues.....
        vgrid_dim = voxelgrid.x_y_z
        flat_pred_class = pred_class[0:scene_end,:].reshape(-1)
        # +1 for freespace and another one for density (amount of points in each voxel)
        vox_prob = np.zeros(shape=(vgrid_dim[0], vgrid_dim[1], vgrid_dim[2], NUM_CLASSES+2))
        # vox_prob = np.zeros(shape=(vgrid_dim[0], vgrid_dim[1], vgrid_dim[2], NUM_CLASSES+1))
        vox_prob[:,:,:, NUM_CLASSES] = 1 #freespace
        voxels = voxelgrid.voxel_n
        filled_vox = np.unique(voxels)
        for curr_vox in filled_vox:
            idx = np.unravel_index(curr_vox, vgrid_dim)
            vox_prob[idx[0],idx[1],idx[2], NUM_CLASSES] = 0 #freespace 
            vox_points = np.where(voxels == curr_vox)[0]
            # add density to datacost 
            vox_prob[idx[0],idx[1],idx[2], NUM_CLASSES+1] = vox_points.size 
            for p in vox_points:
                vox_prob[idx[0],idx[1],idx[2], flat_pred_class[p]]  +=  1 / vox_points.size
        #normalize density
        vox_prob[:,:,:,NUM_CLASSES+1] /= (np.max(vox_prob[:,:,:,NUM_CLASSES+1]))
        

        print('scene %s segmented' % scene_name)
        return vox_prob


stats_dataframe = pd.DataFrame()
def add_stat(data):
    global stats_dataframe
    stats_dataframe = stats_dataframe.append(data, ignore_index=True)


if __name__ == "__main__":
    segment()
    stats_dataframe.to_csv(os.path.join(BASE_DIR, 'results/results.csv'))
    print('Done')




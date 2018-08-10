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
import pc_util as pc
from pyntcloud import PyntCloud


if os.path.exists("/scratch/thesis/HIL"):
    import ptvsd
    ptvsd.enable_attach("thesis", address = ('192.33.89.41', 3000))
    ptvsd.wait_for_attach()

NUM_CLASSES = 21
NUM_POINT = 8192
BATCH_SIZE = 32

TEST_WHOLE_SCENE = dataset.WholeScene(num_classes = NUM_CLASSES)

def segment(model_path):
    with tf.device('/gpu:0'):
        pointclouds_pl, labels_pl, sampleweights_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        batch = tf.Variable(0)
        pred = get_model(NUM_CLASSES, pointclouds_pl, is_training_pl)
        loss = get_loss(pred, labels_pl, sampleweights_pl)
        # pred_softmax = tf.nn.softmax(pred)

    saver = tf.train.Saver() 

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True

    
    with tf.Session(config=config) as sess:
        # Restore variables from model_path
        saver.restore(sess, model_path)
        print("Model restored")

        ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'sampleweights_pl' : sampleweights_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'step': batch}


        
        # eval_whole_scene_one_epoch(sess, ops)
        eval_scene(0, sess, ops)

       
    # ops = {'pointclouds_pl': pointclouds_pl,
    #        'labels_pl': labels_pl,
    #        'is_training_pl': is_training_pl}

    # labeled_cloud = PytnCloud(pd.DataFrame({'x': pts[:, 6], 'y': pts[:, 7], 'z': pts[:, 8], 'label': pred_label[b, :]}))

def eval_whole_scene_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(NUM_CLASSES)]

    print('----')

    test_idxs = np.arange(0, len(TEST_WHOLE_SCENE))
    num_batches = len(TEST_WHOLE_SCENE)

    labelweights = np.zeros(21)
    labelweights_vox = np.zeros(21)

    is_continue_batch = False
    extra_batch_data = np.zeros((0,NUM_POINT,6))
    extra_batch_label = np.zeros((0,NUM_POINT))
    extra_batch_smpw = np.zeros((0,NUM_POINT))

    for batch_idx in range(num_batches):
        if not is_continue_batch:
            batch_data, batch_label, batch_smpw = TEST_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data,extra_batch_data),axis=0)
            batch_label = np.concatenate((batch_label,extra_batch_label),axis=0)
            batch_smpw = np.concatenate((batch_smpw,extra_batch_smpw),axis=0)
        else:
            batch_data_tmp, batch_label_tmp, batch_smpw_tmp = TEST_WHOLE_SCENE[batch_idx]
            batch_data = np.concatenate((batch_data,batch_data_tmp),axis=0)
            batch_label = np.concatenate((batch_label,batch_label_tmp),axis=0)
            batch_smpw = np.concatenate((batch_smpw,batch_smpw_tmp),axis=0)
        if batch_data.shape[0]<BATCH_SIZE:
            is_continue_batch = True
            continue
        elif batch_data.shape[0]==BATCH_SIZE:
            is_continue_batch = False
            extra_batch_data = np.zeros((0,NUM_POINT,6))
            extra_batch_label = np.zeros((0,NUM_POINT))
            extra_batch_smpw = np.zeros((0,NUM_POINT))
        else:
            is_continue_batch = False
            extra_batch_data = batch_data[BATCH_SIZE:,:,:]
            extra_batch_label = batch_label[BATCH_SIZE:,:]
            extra_batch_smpw = batch_smpw[BATCH_SIZE:,:]
            batch_data = batch_data[:BATCH_SIZE,:,:]
            batch_label = batch_label[:BATCH_SIZE,:]
            batch_smpw = batch_smpw[:BATCH_SIZE,:]

        feed_dict = {ops['pointclouds_pl']: batch_data,
                        ops['labels_pl']: batch_label,
                        ops['sampleweights_pl']: batch_smpw,
                        ops['is_training_pl']: is_training}
        step, loss_val, pred_val = sess.run([ops['step'], ops['loss'], ops['pred']],
                                    feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 2) # BxN

        correct = np.sum((pred_val == batch_label) & (batch_label>0) & (batch_smpw>0)) # do not use unknown label
        total_correct += correct
        total_seen += np.sum((batch_label>0) & (batch_smpw>0))
        loss_sum += loss_val
        tmp,_ = np.histogram(batch_label,range(22))
        labelweights += tmp
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label==l) & (batch_smpw>0))
            total_correct_class[l] += np.sum((pred_val==l) & (batch_label==l) & (batch_smpw>0))

        for b in range(batch_label.shape[0]):
            _, uvlabel, _ = pc.point_cloud_label_to_surface_voxel_label_fast(batch_data[b,batch_smpw[b,:]>0,:], np.concatenate((np.expand_dims(batch_label[b,batch_smpw[b,:]>0],1),np.expand_dims(pred_val[b,batch_smpw[b,:]>0],1)),axis=1), res=0.02)
            total_correct_vox += np.sum((uvlabel[:,0]==uvlabel[:,1])&(uvlabel[:,0]>0))
            total_seen_vox += np.sum(uvlabel[:,0]>0)
            tmp,_ = np.histogram(uvlabel[:,0],range(22))
            labelweights_vox += tmp
            for l in range(NUM_CLASSES):
                total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
                total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

            #Output Prediction Pointcloud and truth not correct yet
            scene_data = batch_data[b]
            fout = open(os.path.join(BASE_DIR, 'results', 'predicted_%d_%d.obj' % (batch_idx, b)), 'w+')
            for i in range(scene_data.shape[0]):
                color = label_util.label2color(pred_val[b][i], converted=True)
                fout.write('v %f %f %f %d %d %d\n' % (scene_data[i,0], scene_data[i,1], scene_data[i,2], color[0], color[1], color[2]))
            fout.close()
            fout = open(os.path.join(BASE_DIR, 'results', 'truth_%d_%d.obj' % (batch_idx, b)), 'w+')
            for i in range(scene_data.shape[0]):
                color = label_util.label2color(batch_label[b][i].astype(int), converted=True)
                fout.write('v %f %f %f %d %d %d\n' % (scene_data[i,0], scene_data[i,1], scene_data[i,2], color[0], color[1], color[2]))
            fout.close()
            


       
       

    print('eval whole scene mean loss: %f' % (loss_sum / float(num_batches)))
    print('eval whole scene point accuracy vox: %f'% (total_correct_vox / float(total_seen_vox)))
    print('eval whole scene point avg class acc vox: %f' % (np.mean(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6))))
    print('eval whole scene point avg class acc: %f' % (np.mean(np.array(total_correct_class[1:])/(np.array(total_seen_class[1:],dtype=np.float)+1e-6))))
    print('eval whole scene point accuracy: %f'% (total_correct / float(total_seen)))
    labelweights = labelweights[1:].astype(np.float32)/np.sum(labelweights[1:].astype(np.float32))
    labelweights_vox = labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
    caliweights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    caliacc = np.average(np.array(total_correct_class_vox[1:])/(np.array(total_seen_class_vox[1:],dtype=np.float)+1e-6),weights=caliweights)
    print('eval whole scene point calibrated average acc vox: %f' % caliacc)

    per_class_str = 'vox based --------'
    for l in range(1,NUM_CLASSES):
	    per_class_str += 'class %d weight: %f, acc: %f; \n' % (l,labelweights_vox[l-1],total_correct_class_vox[l]/float(total_seen_class_vox[l]))
    # log_string(per_class_str)


def eval_scene(scene_id, sess, ops):
    is_training = False
    
    print('----')

    batch_data, _, _ = TEST_WHOLE_SCENE[scene_id]

    scene_end = batch_data.shape[0]

    if batch_data.shape[0]<BATCH_SIZE:
        batch_data = np.concatenate((batch_data, np.zeros((BATCH_SIZE-scene_end,NUM_POINT,6))))
    else:
        print("error")

    feed_dict = {ops['pointclouds_pl']: batch_data,
                 ops['is_training_pl']: is_training}
    step,  pred_val = sess.run([ops['step'], ops['pred']],
                                feed_dict=feed_dict)
    pred_val = np.argmax(pred_val, 2) # BxN


    #Output Prediction Pointcloud and truth not correct yet
    fout = open(os.path.join(BASE_DIR, 'results', 'predicted_scene_%d.obj' % scene_id), 'w+')
    fout_voxel = open(os.path.join(BASE_DIR, 'results', 'forvoxel_%d.obj' % scene_id), 'w+')
    for b in range(scene_end):
        for i in range(batch_data.shape[1]):
            color = label_util.label2color(pred_val[b][i], converted=True)
            fout.write('v %f %f %f %d %d %d\n' % (batch_data[b,i,0], batch_data[b,i,1], batch_data[b,i,2], color[0], color[1], color[2]))
            fout_voxel.write('v %f %f %f\n' % (batch_data[b,i,0], batch_data[b,i,1], batch_data[b,i,2]))

    fout.close()
    fout_voxel.close()

    # cloud = PyntCloud.from_file(os.path.join(BASE_DIR, 'results', 'forvoxel_%d.obj' % scene_id))
    # voxelgrid_id = cloud.add_structure("voxelgrid", sizes=[0.1, 0.1, 0.1])
    # voxelgrid = cloud.structures[voxelgrid_id]
    # voxelgrid.plot(d=3, mode="density", cmap="hsv")

            





if __name__ == "__main__":
    segment("/scratch/thesis/segmentation/model/model.ckpt")

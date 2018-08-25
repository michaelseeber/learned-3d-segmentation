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
import pandas as pd



if os.path.exists("/scratch/thesis/HIL"):
    import ptvsd
    ptvsd.enable_attach("thesis", address = ('192.33.89.41', 3000))
    ptvsd.wait_for_attach()

NUM_CLASSES = 21
NUM_POINT = 8192
BATCH_SIZE = 40

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

    # segment pointclouds that have groundtruth for reconstruction
    scene_list = []
    with open(os.path.join('/scratch/thesis/data/scenes/reconstruct_gt', 'list.txt'), "r") as fid:
        for line in fid:
            line = line.strip()
            if line:
                scene_list.append(line)
    
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
        for idx, scene_name in enumerate(scene_list):
            eval_scene(idx, scene_name, sess, ops)

       
    # ops = {'pointclouds_pl': pointclouds_pl,
    #        'labels_pl': labels_pl,
    #        'is_training_pl': is_training_pl}

    # labeled_cloud = PytnCloud(pd.DataFrame({'x': pts[:, 6], 'y': pts[:, 7], 'z': pts[:, 8], 'label': pred_label[b, :]}))


def eval_scene(scene_id, scene_name, sess, ops):
    is_training = False
    
    print('----')

    batch_data, gt_label, _ = TEST_WHOLE_SCENE[scene_id]

    scene_end = batch_data.shape[0]

    if batch_data.shape[0]<BATCH_SIZE:
        batch_data = np.concatenate((batch_data, np.zeros((BATCH_SIZE-scene_end,NUM_POINT,6))))
    else:
        print("error")

    feed_dict = {ops['pointclouds_pl']: batch_data,
                 ops['is_training_pl']: is_training}
    step,  pred_val = sess.run([ops['step'], ops['pred']],
                                feed_dict=feed_dict)
    pred_class = np.argmax(pred_val, 2) # BxN


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

    pandasdata = pd.DataFrame(batch_data[0:scene_end, :, 0:3].reshape(-1, 3), columns=['x','y','z'])

    #todo bounding box
    #p bbox = np.loadtxt("/scratch/thesis/data/scenes/reconstruct_gt/scene0000_00/converted/bbox.txt")
    # minxyz = bbox[:,0]
    # maxxyz = bbox[:,1]
    # pandasmin = pd.DataFrame(minxyz)
    # pandasmax = pd.DataFrame(maxxyz)
    # pandasdata.append(pandasmin)
    # pandasdata.append(pandasmax)
    cloud = PyntCloud(pandasdata)
    # use resolutionof 5 cm, bounding box not regular
    voxelgrid_id = cloud.add_structure("voxelgrid", sizes=[0.05, 0.05, 0.05], bb_cuboid=False)
    voxelgrid = cloud.structures[voxelgrid_id]

    original = PyntCloud.from_file("/scratch/thesis/data/scenes/full/scene0000_00/scene0000_00_vh_clean_2.ply")
    original_grid_id = original.add_structure("voxelgrid", sizes=[0.05, 0.05, 0.05], bb_cuboid=False)
    original_grid = original.structures[original_grid_id]

    color = None 
    pc.extract_mesh_marching_cubes(os.path.join(BASE_DIR, 'results', 'voxelgrid_%s.ply' % scene_name), voxelgrid.get_feature_vector(mode="binary"), color=color)


    # fill voxelgrid with probabilities of points in it ->numerical issues.....
    vgrid_dim = voxelgrid.x_y_z
    flat_pred_class = pred_class[0:scene_end,:].reshape(-1)
    vox_prob = np.zeros(shape=(vgrid_dim[0], vgrid_dim[1], vgrid_dim[2], NUM_CLASSES+1))
    vox_prob[:,:,:, NUM_CLASSES] = 1 #freespace
    voxels = voxelgrid.voxel_n
    filled_vox = np.unique(voxels)
    for curr_vox in filled_vox:
        idx = np.unravel_index(curr_vox, vgrid_dim)
        vox_prob[idx[0],idx[1],idx[2], NUM_CLASSES] = 0 #freespace 
        vox_points = np.where(voxels == curr_vox)[0]
        for p in vox_points:
            vox_prob[idx[0],idx[1],idx[2], flat_pred_class[p]]  +=  1 / vox_points.size
        

    print('test')
    np.savez_compressed(os.path.join('/scratch/thesis/data/segmented/', 'voxelgrid_%s.npz' % scene_name), volume=vox_prob)
    # class_vox = [0 for _ in range(NUM_CLASSES)]
    # _, uvlabel, _ = pc.point_cloud_label_to_surface_voxel_label_fast(batch_data[0:scene_end,:,0:3].reshape(-1, 3), pred_class[0:scene_end,:].reshape(-1), res=0.05)
    # for l in range(NUM_CLASSES):
    #         class_vox[l] += np.sum(uvlabel[:]==l)

    # per_class_str = 'vox based --------'
    # for l in range(0,NUM_CLASSES):
    #     per_class_str += 'class %d amount %d \n' % (l, class_vox[l])
    # print(per_class_str)



if __name__ == "__main__":
    segment("/scratch/thesis/segmentation/model/model.ckpt")

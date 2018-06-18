import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from model import *
from pyntcloud import PyntCloud



def segment(model_path, data_path):

    #TODO INPUT preprocess i.e. center
    input_data = PyntCloud.from_file(os.path.join(data_path, "scene0000_00_vh_clean_2.ply"))
    train_data = input_data.points.drop('alpha', axis=1).values
    input_data.points['x'] = (input_data.points['x'] - input_data.points['x'].mean())
    input_data.points['y'] = (input_data.points['y'] - input_data.points['y'].mean())
    input_data.points['z'] = (input_data.points['z'] - input_data.points['z'].mean())
     
    with tf.device('/gpu:0'):
        pointclouds_pl, _ = placeholder_inputs(BATCH_SIZE)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        #load model
        pred = get_model(pointclouds_pl, tc.constant(False))

    saver = tf.train.Saver() 

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    
    with tf.Session(config=config) as sess:
        # Restore variables from model_path
        saver.restore(sess, model_path)
        log_string("Model restored")

        prediction = sess.run(pred, feed_dict={pointclouds_pl:input_data})

    # ops = {'pointclouds_pl': pointclouds_pl,
    #        'labels_pl': labels_pl,
    #        'is_training_pl': is_training_pl}

    # labeled_cloud = PytnCloud(pd.DataFrame({'x': pts[:, 6], 'y': pts[:, 7], 'z': pts[:, 8], 'label': pred_label[b, :]}))

if __name__ == "__main__":
    segment("", "")

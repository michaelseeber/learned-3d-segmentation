import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from model import *
from pyntcloud import PyntCloud
import label_util



def segment(model_path, data_path):

    #TODO INPUT preprocess i.e. center
    input_data = PyntCloud.from_file(os.path.join(data_path, "scene0000_00_vh_clean_2.ply"))
    input_data.points['x'] = (input_data.points['x'] - input_data.points['x'].mean())
    input_data.points['y'] = (input_data.points['y'] - input_data.points['y'].mean())
    input_data.points['z'] = (input_data.points['z'] - input_data.points['z'].mean())
    input_data = input_data.points.drop('alpha', axis=1).values

    input_size = input_data.shape[0]
     
    with tf.device('/gpu:0'):
        pointclouds_pl, _ = placeholder_inputs(input_size)

        #load model
        pred = get_model(pointclouds_pl, tf.constant(False))
        pred_softmax = tf.nn.softmax(pred)

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

        prediction = sess.run(pred_softmax, feed_dict={pointclouds_pl:input_data})
        pred_labels = np.argmax(prediction, 1)

        print(prediction.shape)

        fout = open(os.path.join(data_path, '_predictedSegmentation.obj'), 'w')
        for i in range(input_size):
            color = label_util.label2color(pred_labels[i])
            fout.write('v %f %f %f %d %d %d\n' % (input_data[i,0], input_data[i,1], input_data[i,2], color[0], color[1], color[2]))
        fout.close()

    # ops = {'pointclouds_pl': pointclouds_pl,
    #        'labels_pl': labels_pl,
    #        'is_training_pl': is_training_pl}

    # labeled_cloud = PytnCloud(pd.DataFrame({'x': pts[:, 6], 'y': pts[:, 7], 'z': pts[:, 8], 'label': pred_label[b, :]}))

if __name__ == "__main__":
    segment("/scratch/thesis/segmentation/model/model.ckpt", "/scratch/thesis/data/scenes/scene0000_00")

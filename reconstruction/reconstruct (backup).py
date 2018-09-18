import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from train_scannet import pointnet_data_generator, build_model, categorical_crossentropy, classification_accuracy


SEG_POINTCLOUDS_PATH = '/scratch/thesis/data/segmented'



def evaluate(datacost, model_path, params):

    batch_size = params["batch_size"]
    nlevels = params["nlevels"]
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]
    nclasses = params["nclasses"]
  

    probs, d, u, u_, m, l = build_model(params)

    u_init = []
    u_init_ = []
    m_init = []
    l_init = []
    for level in range(nlevels):
        factor = 2 ** level
        assert nrows % factor == 0
        assert ncols % factor == 0
        assert nslices % factor == 0
        nrows_level = nrows // factor
        ncols_level = ncols // factor
        nslices_level = nslices // factor
        u_init.append(np.empty([batch_size, nrows_level, ncols_level,
                                nslices_level, nclasses],
                               dtype=np.float32))
        u_init_.append(np.empty([batch_size, nrows_level, ncols_level,
                                 nslices_level, nclasses],
                                dtype=np.float32))
        m_init.append(np.empty([batch_size, nrows_level, ncols_level,
                                nslices_level, 3 * nclasses],
                               dtype=np.float32))
        l_init.append(np.empty([batch_size, nrows_level, ncols_level,
                                nslices_level],
                               dtype=np.float32))

    pred_labels = tf.argmax(probs[0], axis=-1)

    with tf.Session() as sess:
        # Restore variables from model_path
        saver = tf.train.Saver() 
        saver.restore(sess, model_path)
        print("Model restored")

        # sess.run(tf.global_variables_initializer())

        num_batch_samples = datacost.shape[0]

        feed_dict = {}

        feed_dict[d] = datacost

        for level in range(nlevels):
            u_init[level][:] = 1.0 / nclasses
            u_init_[level][:] = 1.0 / nclasses
            m_init[level][:] = 0.0
            l_init[level][:] = 0.0
            feed_dict[u[level]] = u_init[level][:num_batch_samples]
            feed_dict[u_[level]] = u_init_[level][:num_batch_samples]
            feed_dict[m[level]] = m_init[level][:num_batch_samples]
            feed_dict[l[level]] = l_init[level][:num_batch_samples]

        pred = sess.run(
            [pred_labels],
            feed_dict=feed_dict
        )


        return "Hallo"


def reconstruct():

    args = parse_args()

    np.random.seed(0)
    tf.set_random_seed(0)

    tf.logging.set_verbosity(tf.logging.INFO)


    datacost_path = os.path.join(SEG_POINTCLOUDS_PATH,"voxelgrid_" + "scene0000_00" +".npz")
    datacost = np.load(datacost_path)["volume"]

    params = {
        "nlevels": args.nlevels,
        "niter": args.niter,
        "sig": args.sig,
        "tau": args.tau,
        "lam": args.lam,
        "learning_rate": args.learning_rate,
        "softmax_scale": args.softmax_scale,
        "batch_size" : 1,
        "nrows" : datacost.shape[0],
        "ncols" : datacost.shape[1],
        "nslices" : datacost.shape[2],
        "nclasses" : datacost.shape[3],
    }

    probs = evaluate(datacost, args.model_path, params)

    




def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--scene_path", required=True)
    parser.add_argument("--scene_train_list_path", required=True)
    parser.add_argument("--scene_val_list_path", required=True)
    parser.add_argument("--model_path", required=True)

    parser.add_argument("--nclasses", type=int, required=True)

    parser.add_argument("--nlevels", type=int, default=3)

    parser.add_argument("--niter", type=int, default=50)
    parser.add_argument("--sig", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--softmax_scale", type=float, default=10)

    return parser.parse_args()




if __name__ == "__main__":
    reconstruct()

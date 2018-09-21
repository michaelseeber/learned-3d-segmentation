import os
import glob
import argparse
import numpy as np
import tensorflow as tf
from train_scannet import pointnet_data_generator, build_model
from skimage.measure import marching_cubes_lewiner
import plyfile


SEG_POINTCLOUDS_PATH = '/scratch/thesis/data/segmented'



def evaluate(data, model_path, params):
    batch_size = params["batch_size"]
    nlevels = params["nlevels"]
    nclasses = params["nclasses"]
    
    nrows = params["nrows"]
    ncols = params["ncols"]
    nslices = params["nslices"]

    probs, datacost, u, u_, m, l = build_model(params)

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


    with tf.Session() as sess:
        # Restore variables from model_path
        saver = tf.train.Saver() 
        saver.restore(sess, model_path)
        print("Model restored")
            
        feed_dict = {}

        data = data[np.newaxis, ...]
        feed_dict[datacost] = data[:, :nrows, :ncols, :nslices, :]

        for level in range(nlevels):
            u_init[level][:] = 1.0 / nclasses
            u_init_[level][:] = 1.0 / nclasses
            m_init[level][:] = 0.0
            l_init[level][:] = 0.0
            feed_dict[u[level]] = u_init[level][:]
            feed_dict[u_[level]] = u_init_[level][:]
            feed_dict[m[level]] = m_init[level][:]
            feed_dict[l[level]] = l_init[level][:]

        pred = sess.run(
            [tf.argmax(probs[0], axis=-1)],
            feed_dict=feed_dict
        )

        return np.squeeze(pred[0])


            
SEG_POINTCLOUDS_PATH = '/scratch/thesis/data/segmented'
GROUNDTRUTH_PATH = '/scratch/thesis/data/scenes/reconstruct_gt'

def reconstruct():

    args = parse_args()

    scene_name = "scene0000_00"
    datacost_path = os.path.join(SEG_POINTCLOUDS_PATH,"voxelgrid_" + scene_name +".npz")
    datacost_data = np.load(datacost_path)
    datacost = datacost_data["volume"]

    params = {
        "nrows": int(datacost.shape[0] / 4)*4,
        "ncols": int(datacost.shape[1] / 4)*4,
        "nslices": int(datacost.shape[2] / 4)*4,
        "nlevels": 3,
        "batch_size": 1,
        "nclasses": args.nclasses,
        "niter": args.niter,
        "sig": args.sig,
        "tau": args.tau,
        "lam": args.lam,
        "softmax_scale": 10,
    }

    prediction = evaluate(datacost, args.model_path, params)

    prediction[prediction!=21] = 1
    prediction[prediction==21] = 0

    extract_mesh_marching_cubes('/scratch/test.ply', prediction)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--nclasses", type=int, required=True)
    parser.add_argument("--niter", type=int, default=50)
    parser.add_argument("--sig", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=1.0)

    return parser.parse_args()

def extract_mesh_marching_cubes(path, volume, color=None, level=0.5,
                                step_size=1.0, gradient_direction="ascent"):
    if level > volume.max() or level < volume.min():
        print('error')

    verts, faces, normals, values = marching_cubes_lewiner(
        volume, level=level, step_size=step_size,
        gradient_direction=gradient_direction)

    ply_verts = np.empty(len(verts),
                         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    ply_verts["x"] = verts[:, 0]
    ply_verts["y"] = verts[:, 1]
    ply_verts["z"] = verts[:, 2]
    ply_verts = plyfile.PlyElement.describe(ply_verts, "vertex")

    if color is None:
        ply_faces = np.empty(
            len(faces), dtype=[("vertex_indices", "i4", (3,))])
    else:
        ply_faces = np.empty(
            len(faces), dtype=[("vertex_indices", "i4", (3,)),
                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        ply_faces["red"] = color[0]
        ply_faces["green"] = color[1]
        ply_faces["blue"] = color[2]
    ply_faces["vertex_indices"] = faces
    ply_faces = plyfile.PlyElement.describe(ply_faces, "face")

    plyfile.PlyData([ply_verts, ply_faces]).write(path)


if __name__ == "__main__":
    reconstruct()

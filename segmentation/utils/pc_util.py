import os
import sys
import numpy as np
import plyfile
from skimage.measure import marching_cubes_lewiner


def extract_mesh_marching_cubes(path, volume, color=None, level=0.5,
                                step_size=1.0, gradient_direction="ascent"):
    if level > volume.max() or level < volume.min():
        return

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

def point_cloud_label_to_surface_voxel_label_fast(point_cloud, label, res=0.0484):
    coordmax = np.max(point_cloud,axis=0)
    coordmin = np.min(point_cloud,axis=0)
    nvox = np.ceil((coordmax-coordmin)/res)
    vidx = np.ceil((point_cloud-coordmin)/res)
    vidx = vidx[:,0]+vidx[:,1]*nvox[0]+vidx[:,2]*nvox[0]*nvox[1]
    uvidx, vpidx = np.unique(vidx,return_index=True)
    if label.ndim==1:
        uvlabel = label[vpidx]
    else:
        assert(label.ndim==2)
        uvlabel = label[vpidx,:]
    return uvidx, uvlabel, nvox
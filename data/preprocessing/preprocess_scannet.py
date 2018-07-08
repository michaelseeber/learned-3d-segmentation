import os
import json
import socket
import scannet_util
import json
import numpy as np
import pickle
from plyfile import PlyData, PlyElement
BASE_DIR = os.path.dirname(__file__)

    
CLASS_NAMES = scannet_util.g_label_names
RAW2SCANNET = scannet_util.g_raw2scannet


# if os.path.exists("/scratch/thesis/HIL"):
#     import ptvsd
#     ptvsd.enable_attach("thesis", address = ('192.33.89.41', 3000))
#     ptvsd.wait_for_attach()


SCANNET_DIR = '/scratch/thesis/data/scenes'
SCENE_NAMES = [line.rstrip() for line in open(os.path.join(SCANNET_DIR, 'list.txt'))]

def main():
    
    # fetch all scannet labels from scenes - used just for information
    # labels = fetch_classes()

    allpoints = []
    alllabels = []
    for scene_name in SCENE_NAMES:
        print("Processing scene %s" % scene_name)
        #load data
        points, labels = collect_data_one_scene(scene_name)
        #Center PointCloud
        # points[:, 0:3] = points[:, 0:3] - points[:, 0:3].mean(axis=0, keepdims=True)
        allpoints.append(points)
        alllabels.append(labels)
    #save to disk
    with open('data.pickle', 'wb') as f:
        print("Saving to disk...")
        pickle.dump([allpoints, np.squeeze(alllabels)], f)
        print("Done!")





def collect_data_one_scene(scene_name):
    # Over-segmented segments: maps from segment to vertex/point IDs
    data_folder = os.path.join(SCANNET_DIR, scene_name)
    mesh_seg_filename = os.path.join(data_folder, '%s_vh_clean_2.0.010000.segs.json'%(scene_name))
    with open(mesh_seg_filename) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)
    
    # Raw points in XYZRGBA
    ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
    points = read_ply_rgb(ply_filename)
    
    # Instances over-segmented segment IDs: annotation on segments
    instance_segids = []
    labels = []
    annotation_filename = os.path.join(data_folder, '%s.aggregation.json'%(scene_name))
    #print annotation_filename
    with open(annotation_filename) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            instance_segids.append(x['segments'])
            labels.append(x['label'])
    
    # Each instance's points
    instance_points_list = []
    instance_labels_list = []
    semantic_labels_list = []
    for i in range(len(instance_segids)):
       segids = instance_segids[i]
       pointids = []
       for segid in segids:
           pointids += segid_to_pointid[segid]
       instance_points = points[np.array(pointids),:]
       instance_points_list.append(instance_points)
       instance_labels_list.append(np.ones((instance_points.shape[0], 1))*i)   
       if labels[i] not in RAW2SCANNET:
           label = 'unannotated'
       else:
           label = RAW2SCANNET[labels[i]]
       label = CLASS_NAMES.index(label)
       semantic_labels_list.append(np.ones((instance_points.shape[0], 1), dtype=np.int8)*label)
       
    # Refactor data format
    scene_points = np.concatenate(instance_points_list, 0)
    scene_points = scene_points[:,0:6] # XYZRGB, disregarding the A
    instance_labels = np.concatenate(instance_labels_list, 0) 
    semantic_labels = np.concatenate(semantic_labels_list, 0)
    return scene_points, semantic_labels

def read_ply_rgb(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.column_stack((pc['x'], pc['y'],pc['z'], pc['red'],pc['green'],pc['blue']))
    return pc_array


def fetch_classes():
    """fetch all scannet classes from the scenes
    
    Returns:
        list: containing labels
    """

    labels = set()
    for scene_name in SCENE_NAMES:
        path = os.path.join(SCANNET_DIR, scene_name)
        agg_filename = os.path.join(path, scene_name+'.aggregation.json')
        with open(agg_filename) as jsondata:
            d = json.load(jsondata)
            for x in d['segGroups']:
                labels.add(x['label']) 
    print("Extracted %d classes" % len(labels))
    return list(labels)


if __name__ == "__main__":
    main()

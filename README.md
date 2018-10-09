# Bachelor Thesis: Learned 3D Semantic Segmentation

This repository contains the corresponding code of the Bachelor Thesis "Learned 3D Semantic Segmentation". 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. Make sure that you have downloaded the ScanNet dataset or have already preprocessed scene files at hand.

### Overall Structure of this repository

**Data folder**: Contains everthing related to the dataset as well as preprocessing
* dataset.py: Yields blocks or whole scenes from the preprocessed pointclouds. Used by the segmentation network.
* preprocessing folder: included everything needed for preprocessing the ScanNet dataset.

* (scenes) folder: needs to be created by the user. Should contain a subfolder with the whole ScanNetcannet. Furthermore the preprocessed data for segmentation as well as the generated groundtruth for reconstrucion should be in this folder.
* (segmented) folder: should be created by the user. The segmented point cloud gets saved as voxelgrid into this folder by segment.py

**Segmentation folder:** Contains everthing related to the segmentation of pointclouds
* train_segmentation.py: trains PointNet adoption
* segment.py: segments point clouds and saves voxelgrid to disk
* model.py: contains the model
* utils folder: contains helper functions

* (results) folder: Without modification the segment.py saves the visualizations of the segmented point clouds into this folder

**Reconstruction folder**: Contains everything related to reconstruction
* train_scannet.py: generates datacost from voxelgrid and trains reconstruction
* reconstruct.py: can be used to evaluate/reconstruct scenes from a previously trained model.

* (results) folder: Without modifications the reconstruct.py saves the visualizations of the reconstructionsthis folder
    
## Run the code

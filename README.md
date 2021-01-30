<img src="https://github.com/michaelseeber/thesis/blob/master/title_figure.png" height="300">

# Bachelor Thesis: <br /> Learned 3D Semantic Segmentation using Point Clouds

This repository contains the source code of the bachelor thesis "Learned 3D Semantic Segmentation using Point Clouds" by Michael Seeber.

The thesis itself is available [**here**](Bachelor_Thesis_Michael_Seeber.pdf). Additionally [these slides](Presentation_Michael_Seeber.pptx) provide an overview and present the key findings.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. Make sure that you have downloaded the ScanNet dataset or have already preprocessed scene files at hand.

### Overall Structure of this repository

**Data folder**: Contains everthing related to the dataset as well as preprocessing
* `dataset.py`: Yields blocks or whole scenes from the preprocessed point clouds.  Is used by the segmentation network.
* `preprocessing folder`: included everything needed for preprocessing the ScanNet dataset.

* `/scenes` folder: should be created by the user and contain a subfolder with the whole ScanNet dataset. Furthermore the preprocessed data for segmentation as well as the generated groundtruth for reconstrucion should be in this folder.
* `/segmented` folder: should be created by the user. The segmented point cloud gets saved as voxelgrid into this folder by `segment.py`

**Segmentation folder:** Contains everthing related to the segmentation of pointclouds
* `train_segmentation.py`: trains PointNet adoption
* `segment.py`: segments point clouds and saves voxelgrid to disk
* `model.py`: contains the model
* `/utils folder`: contains helper functions

* `/results` folder: Without modification the segment.py saves the visualizations of the segmented point clouds into this folder

**Reconstruction folder**: Contains everything related to reconstruction
* `train_scannet.py`: generates datacost from voxelgrid and trains reconstruction network
* `reconstruct.py`: can be used to evaluate/reconstruct scenes from a previously trained model.

* `/results folder`: Without modifications `reconstruct.py` saves the visualizations of the reconstructionsthis into this folder
    
## Run the code

### Segmentation Part:
* Run `preprocessing/preprocess_scannet.py` to perpare .pickle files that contain all the information used by the segmentation part. Don't forget to modify the path to the ScanNet directory. The file requires a list of scenes, which should be preprocessed, as parameter. Continue once you have a train.pickle file containing the training scenes and a test.pickle file containing the test scenes.

* Modify the the configuration variables inside `segmentation/train_segmentation.py` to match your folder setup. Then run this file to train the segmentation network.

* When training has finished, you can execute `segmentation/segment.py` (once again adjust the configuration variables) to create the voxelgrids.

### Reconstruction Part:
* To create the grountruth for the reconstruction follow the steps noted inside `preprocessing/reconstruction/info.txt`.
* Once completed, you have the segmented voxelgrids as well as the groundtruth reconstructions inside your `data` folder. Therefore the last step is to adjust the configuration variables in `reconstruction/train_scannet.py` and afterwards run it. A sample run command with all the required parameters can be found in `reconstruction/run.txt`
* After training simply run `reconstruction/reconstruct.py` to obtain the reconstruction results from our thesis.



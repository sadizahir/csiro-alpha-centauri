# mri-research
Summer vacation work on MRI/CT analysis at CSIRO.

## Dependencies
numpy, matplotlib, sklearn, skimage.

## Usage
### First Option (Deprecated)
1. Install dill and joblib.
2. Configure the 3D MRI/CT folders.
3. Set path constants in classifier_ct.py
4. Uncomment functions of interest in main.
5. Run classifier_ct.py, where the first argument is number of processes to use.

### Second Option (Recommended)
Run classifier_cluster.py with the following optional arguments:

-t:
Generate (t)raining data using the MRI volumes inside the "path" directory. The
training data will be stored as a *.pkl file which can be re-used for testing
and segmentation reconstruction efforts.

-l:
Reconstruct all the (l)abels found in the training data *.pkl file generated
above and put the reconstructions in a directory.

-r:
Go through all the reconstructions and make a (r)eport comparing the
reconstructions with the ground truth labels stored in the training data *.pkl
file generated above and save the reports in a directory.

### Third Option: 3D (in-progress)
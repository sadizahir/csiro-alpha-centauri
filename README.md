# MRI and CT Research
Summer vacation work on MRI/CT analysis at CSIRO.

## Dependencies
numpy, matplotlib, sklearn, skimage, dill, joblib.

## Usage
### First Option (Deprecated)
1. Configure the 3D MRI/CT folders.
2. Set path constants in classifier_ct.py
3. Uncomment functions of interest in main.
4. Run classifier_ct.py, where the first argument is number of processes to use.

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
Coming soon!

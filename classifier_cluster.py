# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:16:35 2016

@author: Sadi
"""

"""
Commandline arguments for classification routine:

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
"""
# -*- coding: utf-8 -*-
"""
Code to follow the dicom introductory tutorial. Will read some DICOMs.

Created on Tue Dec 01 08:07:19 2015

@author: ZAH010
"""

# Packages
import dicom
import os
import numpy as np
from matplotlib import pyplot, cm

# Path to the DICOMs we're interested in
PathDicom = "../Crohns/converted/anon_1181403/anon/tgz/anon_mr_150420_30sec"

# Store the paths to the entire DICOM series
FileListDicom = [] # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if filename.lower().endswith(".dcm"): # if it's a DICOM extension
            FileListDicom.append(os.path.join(dirName, filename))

# Note: For the DICOMs to be sorted correctly, the filename numbers need to
# have leading zeroes (i.e. 0009, 0010 and not 9, 10)

# Read the first file; this is the "reference" DICOM so that we know what
# the dimensions and attributes of the rest of the DICOMs are
RefDs = dicom.read_file(FileListDicom[0])

# Get the dimensions of the file
PixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(FileListDicom))

# Get the spacing values (in mm) (?)
PixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), 
                float(RefDs.SliceThickness))

# axes for display                
x = np.arange(0.0, (PixelDims[0]+1)*PixelSpacing[0], PixelSpacing[0])
y = np.arange(0.0, (PixelDims[1]+1)*PixelSpacing[1], PixelSpacing[1])
z = np.arange(0.0, (PixelDims[2]+1)*PixelSpacing[2], PixelSpacing[2])

# Create the 3D array to contain all the image data
# The size is something we know from PixelDims
ArrayDicom = np.zeros(PixelDims, dtype=RefDs.pixel_array.dtype)

# Go through all the DICOM files and store them in the array
for index, filename in enumerate(FileListDicom):
    ds = dicom.read_file(filename) # read the file
    ArrayDicom[:, :, index] = ds.pixel_array # store the image data

pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(y, x, np.flipud(ArrayDicom[:, :, len(FileListDicom)/2]))
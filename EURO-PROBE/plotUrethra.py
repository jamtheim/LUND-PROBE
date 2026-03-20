import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.colors import LinearSegmentedColormap, Normalize, hex2color
from matplotlib.cm import get_cmap
import math
# Import plot functions
from ioPlotFunctions import plotFunctions
# Initialize plot functions
plotFunctions = plotFunctions()

from utils_urethra import get_available_delineators, validate_delineator


############### START CONFIGURATION BLOCK ###############
# Define the base path
basePath = r'...\lund-probe\basePart'
# Define selected patient 
patient = "newAcq_4da6c8ac3efcf52d"
# Define the full subject directory
subjectDir = os.path.join(basePath, patient)
# Define name of the MRI directory used in the data
MRIDir = 'MR_StorT2'
EuroProbeDir = 'euro_probe'

# ------------------------------------------------------------------
# Detect available delineators for this case
# ------------------------------------------------------------------
available_delineators = get_available_delineators(os.path.join(subjectDir, MRIDir, EuroProbeDir), verbose=True)
# Select delineator
delineator = 'rad_2'  # Change this to select a different delineator
# Validate delineator
validate_delineator(delineator, available_delineators)

# ------------------------------------------------------------------
# Define the paths to the MRI and EURO-PROBE segmentations
# ------------------------------------------------------------------
MRIVolumePath = os.path.join(subjectDir, MRIDir, 'image.nii.gz')
EuroProbeProstatePath = os.path.join(subjectDir, MRIDir, EuroProbeDir, f'mask_prostate_{delineator}.nii.gz')
EuroProbeUrethraPath = os.path.join(subjectDir, MRIDir, EuroProbeDir, f'mask_urethra_{delineator}.nii.gz')
############### END CONFIGURATION BLOCK ###############

# Load the volumes using SimpleITK
# Image volumes
MRIImage = sitk.ReadImage(MRIVolumePath)
# EURO-PROBE segmentations
MRIProstateImage = sitk.ReadImage(EuroProbeProstatePath)
MRIUrehtraImage = sitk.ReadImage(EuroProbeUrethraPath)

# Get the numpy arrays of the images from the SimpleITK objects
# Image volumes
MRIArray = sitk.GetArrayFromImage(MRIImage)
# EURO-PROBE segmentations
MRIProstateArray = sitk.GetArrayFromImage(MRIProstateImage)
MRIUrethraArray = sitk.GetArrayFromImage(MRIUrehtraImage)

# Get the middle slice of the MRI volume (slices first in the numpy array)
middleSliceMRI = MRIArray.shape[0] // 2

# Plot MRI for the middle slice
plt.imshow(MRIArray[middleSliceMRI, :, :], cmap='gray')
plt.title('MRI (middle slice)')
# Add comment below on patient orientation
plt.text(0, 45, 'Left in images is patient right side', color='white', fontsize=12, fontweight='bold')
plt.axis('off')
plt.show()

# Plot the MRI Urethra segmentation for the middle slice
plt.imshow(MRIUrethraArray[middleSliceMRI, :, :], cmap='gray')
plt.title('MRI Urethra segmentation binary mask')
plt.axis('off')
plt.show()

# Plot the MRI and the Prostate + Urethra segmentations
plotFunctions.plotData(MRIArray, structUrethraArray=MRIUrethraArray, structProstateArray=MRIProstateArray, showDosemap=False, showStructArrays=True, Title=f"MRI with Prostate and Urethra segmentation overlay - Euro-Probe delineator: {delineator}", zooming=True, zoomingShape=[50, 960, 800, 200])
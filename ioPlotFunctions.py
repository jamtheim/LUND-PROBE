import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.colors import LinearSegmentedColormap, Normalize, hex2color
from matplotlib.cm import get_cmap
import math

class plotFunctions():
    def __init__(self):
        pass


    # Functions needed for visualizing radiotherapy images and structures

    def createCustomColormap(self, doseMap, prescribedDose):
        """
        Arg:
            doseMap (np.array): The dose map array
            prescribedDose (float): The prescribed dose in Gy
        Returns:
            doseArray (np.array): The dose map array with values below 10% of prescribedDose masked to nan (for visualization purpose ONLY!)
            eclipse_cmap (matplotlib.colors.LinearSegmentedColormap): The custom colormap to mimic Eclipse colormap
            norm (matplotlib.colors.Normalize): The normalization of the colormap
        """
        doseArray = doseMap.copy()
        doseColorSteps = [
            0.1 * prescribedDose,  # 10% 
            0.3 * prescribedDose,  # 30% 
            0.4 * prescribedDose,  # 40% 
            0.6 * prescribedDose,  # 60% 
            0.8 * prescribedDose,  # 80% 
            (np.max(doseArray)/prescribedDose)* prescribedDose,  # max %
        ]
        normalizedPositions = [(dose - doseColorSteps[0]) / (doseColorSteps[-1] - doseColorSteps[0]) for dose in doseColorSteps]
        # Ensure normalized_positions starts with 0 and ends with 1
        normalizedPositions[0] = 0.0
        normalizedPositions[-1] = 1.0
        # Create the custom colormap
        doseArray[doseArray < doseColorSteps[0]] = np.nan  # Mask values below 10% of prescribedDose

        # Approximate Eclipse colors (replace with actual RGB/hex values if known)
        colors = [
            '#0000FF',  # Blue
            '#0165fc',  # Light Blue
            '#00FFFF',  # Cyan
            '#00FF00',  # Green
            '#FFFF00',  # Yellow
            '#FF0000',  # Red
            ]
        norm = Normalize(vmin=doseColorSteps[0], vmax=doseColorSteps[-1])
        eclipse_cmap = LinearSegmentedColormap.from_list('Eclipse', list(zip(normalizedPositions, colors)))
        
        return doseArray, eclipse_cmap, norm


    def createCustomColormapPET(self, uncertaintyMapArray):
        """
        Arg:
            uncertaintyMapArray (np.array): The uncertainty map array
        Returns:
            uncertaintyMap (np.array): The uncertainty map array
            eclipse_PET_cmap (matplotlib.colors.LinearSegmentedColormap): The custom colormap to mimic Eclipse colormap for PET (Rainbow)
            normPET (matplotlib.colors.Normalize): The normalization of the colormap
        """
        uncertaintyMap = uncertaintyMapArray
        doseColorSteps = [
        0.1 * np.max(uncertaintyMap),
        0.2 * np.max(uncertaintyMap),
        0.3 * np.max(uncertaintyMap),
        0.4 * np.max(uncertaintyMap),
        0.5 * np.max(uncertaintyMap),
        0.7 * np.max(uncertaintyMap),
        1 * np.max(uncertaintyMap),  
        ]
        
        normalizedPositions = [(dose - doseColorSteps[0]) / (doseColorSteps[-1] - doseColorSteps[0]) for dose in doseColorSteps]
        # Ensure normalized_positions starts with 0 and ends with 1
        normalizedPositions[0] = 0.0
        normalizedPositions[-1] = 1.0

        # Approximate Eclipse colors (replace with actual RGB/hex values if known)
        colors = [
            '#000080',  # Dark Blue
            '#0000FF',  # Blue
            '#0165fc',  # Light Blue
            '#00FF00',  # Green
            '#00FFFF',  # Cyan
            '#FFFF00',  # Yellow
            '#FF0000',  # Red  
            ]
        normPET = Normalize(vmin=doseColorSteps[0], vmax=doseColorSteps[-1])
        eclipse_PET_cmap = LinearSegmentedColormap.from_list('Eclipse', list(zip(normalizedPositions, colors)))
        
        return uncertaintyMap, eclipse_PET_cmap, normPET


    def setVmaxVmin(self, imageArray, ct = False):
        """
        Arg:
            imageArray (np.array): The image array
            ct (bool): If the image is a CT image if yes setting fixed vmin and vmax as -160 and 200 like in Eclipse
        Returns:
            vmin (float): The minimum value for the colormap
            vmax (float): The maximum value for the colormap
        """
        if ct:
            vmin = -160
            vmax = 200
        else:
            vmin = np.percentile(imageArray, 0.1)
            vmax = np.percentile(imageArray, 99.9)
        return vmin, vmax


    def plotData(self, imageArray, structRectumArray = None, structCTVArray = None, structPTVArray = None, structFemoralHeadLArray = None, structFemoralHeadRArray = None, 
                structBladderArray = None, structFiducialArray = None, doseMapArray = None, MRICTVObsDImage = None, uncertaintyMapArray = None, showUncertainty = False,
                MRICTVObsCImage = None, showDosemap = False, showStructArrays = False , slice_ax = None, size = 30, ct = False, Title = 'Set title', showLegend = False, zooming = False, zoomingShape = None):
        """
        Arg:
        
            imageArray (np.array): The image array - CT or MRI (Necessary for plotting)
            structRectumArray (np.array): The rectum structure array - If none no contour will be plotted
            structCTVArray (np.array): The CTV structure array - If none no contour will be plotted
            structPTVArray (np.array): The PTV structure array - If none no contour will be plotted
            structFemoralHeadLArray (np.array): The left femoral head structure array - If none no contour will be plotted
            structFemoralHeadRArray (np.array): The right femoral head structure array - If none no contour will be plotted
            structBladderArray (np.array): The bladder structure array - If none no contour will be plotted
            structFiducialArray (np.array): The fiducial structure array - If none no contour will be plotted
            doseMapArray (np.array): The dose map array
            MRICTVObsDImage (np.array): The MRI CTV Obs D image array - If none no contour will be plotted
            MRICTVObsCImage (np.array): The MRI CTV Obs C image array - If none no contour will be plotted
            uncertaintyMapArray (np.array): The uncertainty map array 
            showUncertainty (bool): If the uncertainty map should be shown - If True the uncertainty map will be shown
            showDosemap (bool): If the dose map should be shown - If True the dose map will be shown
            showStructArrays (bool): If the structure arrays should be shown - If True the structure arrays will be shown (If available)
            slice_ax (int): The slice index for the axial view - If None the middle slice will be shown
            size (int): The size of the figure - Default is 30
            ct (bool): If the image is a CT image - If True the vmin and vmax will be set to -160 and 200
            Title (str): The title of the figure - Default is 'Set title'
            showLegend (bool): If the legend should be shown - If True the legend will be shown for contours
            zooming (bool): If the zooming should be applied - If True the zooming will be applied
            zoomingShape (list): The shape of the zooming - Default is [440, 580, 560, 420]
        Return:
            Plot of the gives inputs
        """
        # Get the shape of the image array
        shape = np.shape(imageArray)
        # Define the slice indices for the sagittal, coronal and axial views
        slice_sag = shape[2]//2
        slice_cor = shape[1]//2
        if slice_ax == None:
            slice_ax = shape[0]//2
        # Define the length of the sagittal, coronal and axial views
        sag_len = slice_sag*2
        cor_len = slice_cor*2
        ax_len = slice_ax*2
        # Define the size of the figure
        height_ratios = [cor_len/cor_len]
        x_len = (sag_len/cor_len)*3
        y_len = np.array(height_ratios).sum()/x_len
        gridspec_kw={'width_ratios':[1],'height_ratios':height_ratios}
        # Set vmin and vmax
        vminImage, vmaxImage = self.setVmaxVmin(imageArray, ct)
        # Create the figure and axes
        fig,ax = plt.subplots(1,1,figsize=(size,y_len*size),gridspec_kw=gridspec_kw)
        # Plot the sagittal view of the ImageArray
        ax.imshow(imageArray[slice_ax,:,:], cmap=plt.cm.gray, vmin=vminImage, vmax=vmaxImage, interpolation='spline36')
        # plot the contours of the structures
        if showStructArrays:
            if structRectumArray is not None:
                ax.contour(structRectumArray[slice_ax,:,:], levels=[0], origin="lower", linewidths=2, colors="brown", zorder=1)
            if structCTVArray is not None:
                ax.contour(structCTVArray[slice_ax,:,:], levels=[0], origin="lower", linewidths=2, colors="pink", zorder=1)
            if structPTVArray is not None:
                ax.contour(structPTVArray[slice_ax,:,:], levels=[0], origin="lower", linewidths=2, colors="blue", zorder=1)
            if structFemoralHeadLArray is not None:
                ax.contour(structFemoralHeadLArray[slice_ax,:,:], levels=[0], origin="lower", linewidths=2, colors="yellow", zorder=1)
            if structFemoralHeadRArray is not None:
                ax.contour(structFemoralHeadRArray[slice_ax,:,:], levels=[0], origin="lower", linewidths=2, colors="yellow", zorder=1)
            if structBladderArray is not None:
                ax.contour(structBladderArray[slice_ax,:,:], levels=[0], origin="lower", linewidths=2, colors="yellow", zorder=1)
            if structFiducialArray is not None:
                ax.contour(structFiducialArray[slice_ax,:,:], levels=[0], origin="lower", linewidths=2, colors="red", zorder=1)
            if MRICTVObsDImage is not None:
                ax.contour(MRICTVObsDImage[slice_ax,:,:], levels=[0], origin="lower", linewidths=2, colors="red", zorder=1)
            if MRICTVObsCImage is not None:
                ax.contour(MRICTVObsCImage[slice_ax,:,:], levels=[0], origin="lower", linewidths=2, colors="green", zorder=1)

            if showLegend:
                # Set the legend
                legend_elements = [plt.Line2D([0], [0], marker='_', color='pink', label='CTV', markerfacecolor="pink", markersize=20),
                            plt.Line2D([0], [0], marker='_', color='blue', label='PTV', markerfacecolor="blue", markersize=20),
                            plt.Line2D([0], [0], marker='_', color="brown", label='Rectum', markerfacecolor="brown", markersize=20),
                            plt.Line2D([0], [0], marker='_', color="yellow", label='FemoralHeads', markerfacecolor="yellow", markersize=20)]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=15)

        if showUncertainty:
            uncertaintyMapArrayNorm, cmapCTVUNC, normCTV = self.createCustomColormapPET(uncertaintyMapArray)
            ax.imshow(uncertaintyMapArrayNorm[slice_ax,:,:], cmap=cmapCTVUNC, norm=normCTV, alpha=0.4, interpolation='spline36', zorder=2)
            
        if showDosemap:
            doseMapArrayNorm, eclipse_cmap, norm = self.createCustomColormap(doseMapArray, 42.7)
            ax.imshow(doseMapArrayNorm[slice_ax,:,:], cmap=eclipse_cmap, norm=norm, alpha=0.4, interpolation='spline36', zorder=2)
            #plot colorbar for dose map
            levels = [4.3, 42.7*0.2, 42.7*0.3, 42.7*0.4, 42.7*0.5, 42.7*0.6, 42.7*0.7, 42.7*0.8, 42.7*0.9, 42.7]

            # Process the list
            rounded_levels = [f"{math.ceil(level * 10) / 10:.1f}" for level in levels]

            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=eclipse_cmap), ticks=levels, ax=ax, orientation='vertical')
            cbar.set_label('Dose', fontsize=15, color='white')
            cbar.set_ticklabels(rounded_levels, color="white", fontsize=9)
        if zooming:
            ax.axis(zoomingShape)
            

        # Post processing to make the figure look better
        ax.grid(False)
        # Add title to the figure
        ax.set_title(Title, fontsize=20, color = 'white')
        ax.axis('off')
        fig.subplots_adjust(wspace=0,hspace=0)
        fig.tight_layout()
        fig.patch.set_facecolor('black')
        #fig.show() # Removed to avoid error in the notebook
        
        
    def plotMRIandsCT(self, MRIArray, sCTReg2MRIArray, size=30, Title='Set title', slice_ax=None, zooming = False, zoomingShape = None):
        """
        Arg:
            MRIArray (np.array): The MRI image array
            sCTReg2MRIArray (np.array): The sCT registered to MRI image array
            size (int): The size of the figure - Default is 30
            Title (str): The title of the figure - Default is 'Set title'
            slice_ax (int): The slice index for the axial view - If None the middle slice will be shown
        Return:
            Plot of the MRI and sCT registered in a checkered pattern
        """
        # Get the shape of the image array
        shape = np.shape(MRIArray)
        # Define the slice indices for the sagittal, coronal and axial views
        slice_sag = shape[2]//2
        slice_cor = shape[1]//2
        if slice_ax == None:
            slice_ax = shape[0]//2
        # Define the length of the sagittal, coronal and axial views
        sag_len = slice_sag*2
        cor_len = slice_cor*2
        ax_len = slice_ax*2
        # Define the size of the figure
        height_ratios = [cor_len/cor_len]
        x_len = (sag_len/cor_len)*3
        y_len = np.array(height_ratios).sum()/x_len
        gridspec_kw={'width_ratios':[1],'height_ratios':height_ratios}
        # set vmin and vmax
        vminImage, vmaxImage = self.setVmaxVmin(MRIArray, False)
        # Create the figure and axes
        fig,ax = plt.subplots(1,1,figsize=(size,y_len*size),gridspec_kw=gridspec_kw)
        h, w = shape[-2:]

        ct_top_left = sCTReg2MRIArray[slice_ax,:,:][:h//2, :w//2]
        ct_bottom_right = sCTReg2MRIArray[slice_ax,:,:][h//2:, w//2:]
        mri_top_right = MRIArray[slice_ax,:,:][:h//2, w//2:]
        mri_bottom_left = MRIArray[slice_ax,:,:][h//2:, :w//2]

        composite_image = np.zeros((h, w, 3))

        # Define vmin and vmax for CT and MRI
        ct_vmin, ct_vmax = -160, 200  # Adjust based on your CT scan data range
        mri_vmin, mri_vmax = vminImage, vmaxImage  # Adjust based on your MRI scan data range

        # Apply colormaps with vmin and vmax (without clipping the data)
        #grey_cmap = get_cmap('gray') 
        grey_cmap = matplotlib.colormaps['gray']

        # Top-left: CT with bone colormap
        composite_image[:h//2, :w//2, :] = grey_cmap((ct_top_left - ct_vmin) / (ct_vmax - ct_vmin))[:, :, :3]
        # Bottom-right: CT with bone colormap
        composite_image[h//2:, w//2:, :] = grey_cmap((ct_bottom_right - ct_vmin) / (ct_vmax - ct_vmin))[:, :, :3]
        # Top-right: MRI with viridis colormap
        composite_image[:h//2, w//2:, :] = grey_cmap((mri_top_right - mri_vmin) / (mri_vmax - mri_vmin))[:, :, :3]
        # Bottom-left: MRI with viridis colormap
        composite_image[h//2:, :w//2, :] = grey_cmap((mri_bottom_left - mri_vmin) / (mri_vmax - mri_vmin))[:, :, :3]

        ax.imshow(composite_image, cmap='gray', interpolation='spline36')
        
        if zooming:
            ax.axis(zoomingShape)

        # Post processing to make the figure look better
        ax.grid(False)
        #add title to the figure
        ax.set_title(Title, fontsize=20, color = 'white')
        ax.axis('off')
        fig.subplots_adjust(wspace=0,hspace=0)
        fig.tight_layout()
        fig.patch.set_facecolor('black')
        #fig.show() # Removed to avoid error in the notebook

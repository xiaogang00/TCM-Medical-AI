# encoding: utf-8
import SimpleITK as sitk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import glob

img = sitk.ReadImage('./train_subset/LKDS-00148.mhd')
img_array = sitk.GetArrayFromImage(img)
direction = img.GetDirection()
list(direction)
voxelCoord = np.array([1, 2, 3])
origin = np.array([2, 2, 2])
spacing = np.array([1, 2, 3])


def voxelToWorldCoord(voxelCoord, origin, spacing, direction):
    stretchedWorldCoord = voxelCoord * spacing
    dx = direction[0]
    dy = direction[4]
    dz = direction[8]
    dxyz = [dx, dy, dz]
    worldCoord = stretchedWorldCoord + dxyz * origin
    return worldCoord

voxelToWorldCoord(voxelCoord, origin, spacing, direction)

img_bitmap = Image.fromarray(img_array[100])

plt.imshow(img_bitmap, cmap="gray")
plt.axis("off")
plt.show()

mhds = glob.glob("./train_subset00/*.mhd")
slices = [sitk.ReadImage(x) for x in mhds]
print slices
slices[0].GetDirection()


def make_mask(center, diam, z, width, height, spacing,origin): #只显示结节
    '''
        Center : 圆的中心 px -- list of coordinates x,y,z
        diam : 圆的直径 px -- diameter
        widthXheight : pixel dim of image
        spacing = mm/px conversion rate np array x,y,z
        origin = x,y,z mm np.array
        z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5])
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

make_mask(img.Ge)



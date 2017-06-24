# -*-coding: utf-8 -*-
from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
import scipy.ndimage
from tqdm import tqdm
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import warnings
warnings.filterwarnings("ignore")


def make_mask(center, diam, z, width, height, spacing, origin):  # 只显示结节

    # center : 圆的中心 px -- list of coordinates x,y,z
    # diam : 圆的直径 px -- diameter
    # width and height : pixel dim of image
    # spacing = mm/px conversion rate np array x,y,z
    # origin = x,y,z mm np.array
    # z = z position of slice in world coordinates mm
    mask = np.zeros([height, width])  # 0's everywhere except nodule swapping x,y to match img
    # convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center - origin) / spacing
    v_diam = int(diam / spacing[0] + 5)
    v_xmin = np.max([0, int(v_center[0] - v_diam) - 5])
    v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + 5])
    v_ymin = np.max([0, int(v_center[1] - v_diam) - 5])
    v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + 5])

    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    # Convert back to world coordinates for distance calculation
    x_data = [x * spacing[0] + origin[0] for x in range(width)]
    y_data = [x * spacing[1] + origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
    return (mask)


def matrix2int16(matrix):
    # matrix must be a numpy array NXN
    # Returns uint16 version
    m_min = np.min(matrix)
    m_max = np.max(matrix)
    matrix = matrix - m_min
    return (np.array(np.rint((matrix - m_min) / float(m_max - m_min) * 65535.0), dtype=np.uint16))


# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if ('./sample_patients/' + case + '.mhd') in f:
            return (f)


# The locations of the nodes
def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    """数据标准化"""
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image
    # ---数据标准化


def set_window_width(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    image[image > MAX_BOUND] = MAX_BOUND
    image[image < MIN_BOUND] = MIN_BOUND
    return image
    # ---设置窗宽


if __name__ == "__main__":
    luna_path = './'
    luna_subset_path = luna_path + 'sample_patients/'
    file_list = glob(luna_subset_path + "*.mhd")
    df_node = pd.read_csv(luna_path + 'csv_files/train/' + 'annotations.csv')
    output_path = luna_path + 'npy/'
    working_path = luna_path + 'output/'

    if os.path.isdir(luna_path + '/npy'):
        pass
    else:
        os.mkdir(luna_path + '/npy')

    if os.path.isdir(luna_path + '/output'):
        pass
    else:
        os.mkdir(luna_path + '/output')

    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()
    # print(df_node.head(5))

    print('数据条数：%d' % df_node.shape[0])
    # Tqdm 是一个快速，可扩展的Python进度条，
    for fcount, img_file in enumerate(tqdm(file_list)):
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        # 需要在这里找到指定的file
        if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
            # load the data once，这个df 是含有我们需要的图片的结点的
            itk_img = sitk.ReadImage(img_file)
            # 可以使用sitk这个软件包来读取较大的医学图像数据
            img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
            # 在这里可以用这个函数来得到z, y, x的
            num_z, height, width = img_array.shape  # height and width constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            # 找到原点，
            spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
            # 找到可以进行转换的尺寸
            # go through all nodes (why just the biggest?)
            for node_idx, cur_row in mini_df.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                # just keep 3 slices
                imgs = np.ndarray([3, height, width], dtype=np.float32)
                masks = np.ndarray([3, height, width], dtype=np.uint8)
                center = np.array([node_x, node_y, node_z])  # nodule center
                v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
                for i, i_z in enumerate(np.arange(int(v_center[2]) - 1,
                                                  int(v_center[2]) + 2).clip(0, num_z - 1)):
                    # clip prevents going out of bounds in Z
                    # 在这里只是保存了三个slice，所以v_center在这里从-1到1
                    mask = make_mask(center, diam, i_z * spacing[2] + origin[2],
                                     width, height, spacing, origin)
                    # 在这里的z也需要先经过spacing的变化
                    masks[i] = mask
                    imgs[i] = img_array[i_z]

                np.save(os.path.join(output_path, "images_%04d_%04d.npy" % (fcount, node_idx)), imgs)
                np.save(os.path.join(output_path, "masks_%04d_%04d.npy" % (fcount, node_idx)), masks)

            f, plots = plt.subplots(11, 10, sharex='all', sharey='all', figsize=(10, 11))
            for i in range(110):
                plots[i // 10, i % 10].axis('off')
                plots[i // 10, i % 10].imshow(img_array[i], cmap=plt.cm.bone)
                plt.show()
            fig = plt.figure()
            anim = plt.imshow(img_array[0], cmap=plt.cm.bone)
            plt.show()

            #def update(i):
            #    anim.set_array(img_array[i])
            #    return anim,
            # 在这里使用animation的目的是用来绘制动态图
            # HTML(animation.FuncAnimation(fig, update, frames=range(len(img_array)),
            #                             interval=50, blit=True).to_html5_video())

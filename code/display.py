# -*-coding: utf-8 -*-
from Data import *

if __name__ == '__main__':
    imgs = np.load(os.path.join(output_path, "images_%04d_%04d.npy" % (fcount, node_idx)))
    masks = np.load(os.path.join(output_path, "masks_%04d_%04d.npy" % (fcount, node_idx)))
    # 我们查看保存下来的三层的样子
    for i in range(len(imgs)):
        print ("图片的第 %d 层" % i)
        fig, ax = plt.subplots(2, 2, figsize=[8, 8])
        ax[0, 0].imshow(imgs[i])
        ax[0, 0].set_title(u'彩色切片')
        ax[0, 1].imshow(imgs[i], cmap='gray')
        ax[0, 1].set_title(u'黑白切片')
        ax[1, 0].imshow(masks[i], cmap='gray')
        ax[1, 0].set_title(u'节点')
        ax[1, 1].imshow(imgs[i] * masks[i], cmap='gray')
        ax[1, 1].set_title(u'节点切片')
        plt.show()
        print ('\n\n')
        #    raw_input("hit enter to cont : ")

    # 第二种数据处理方式（2D + 1切片+增强）
    for fcount, img_file in enumerate(tqdm(file_list)):
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
            # load the data once
            itk_img = sitk.ReadImage(img_file)
            img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
            num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
            origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
            # go through all nodes (why just the biggest?)
            for node_idx, cur_row in mini_df.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]

                w_nodule_center = np.array([node_x, node_y, node_z])  # 世界空间中结节中心的坐标
                v_nodule_center = np.rint((w_nodule_center - origin) / spacing)  # 体素空间中结节中心的坐标 (still x,y,z ordering)
                # np.rint 对浮点数取整，但不改变浮点数类型
                # for i, i_z in enumerate(np.arange(int(v_nodule_center[2]) - 1,int(v_nodule_center[2]) + 2).clip(0,num_z - 1)):  # clip 方法的作用是防止超出切片数量的范围
                i_z = int(v_nodule_center[2])
                nodule_mask = make_mask(w_nodule_center, diam, i_z * spacing[2] + origin[2], width, height, spacing,
                                        origin)
                nodule_mask = scipy.ndimage.interpolation.zoom(nodule_mask, [0.5, 0.5], mode='nearest')
                nodule_mask[nodule_mask < 0.5] = 0
                nodule_mask[nodule_mask > 0.5] = 1
                nodule_mask = nodule_mask.astype('int8')
                slice = img_array[i_z]
                slice = scipy.ndimage.interpolation.zoom(slice, [0.5, 0.5], mode='nearest')
                slice = 255.0 * normalize(slice)
                slice = slice.astype(np.uint8)  # ---因为int16有点大，我们改成了uint8图（值域0~255）

                np.save(os.path.join(output_path,
                                     "another/%s_%04d_%04d_%04d.npy" % (cur_row["seriesuid"], fcount, node_idx, i_z)),
                        slice)
                np.save(os.path.join(output_path, "another/%s_%04d_%04d_%04d_mask.npy" % (
                cur_row["seriesuid"], fcount, node_idx, i_z)), nodule_mask)

            fig, ax = plt.subplots(1, 2, figsize=[8, 8])

            aa = np.load(os.path.join(output_path,
                                      "another/%s_%04d_%04d_%04d.npy" % (cur_row["seriesuid"], fcount, node_idx, i_z)))
            bb = np.load(os.path.join(output_path, "another/%s_%04d_%04d_%04d_mask.npy" % (
            cur_row["seriesuid"], fcount, node_idx, i_z)))

            ax[0].imshow(aa)
            ax[0].set_title(u'病例' + cur_row["seriesuid"][-4:] + u'的CT扫描')
            ax[1].imshow(bb)
            ax[1].set_title(u'结节位置')
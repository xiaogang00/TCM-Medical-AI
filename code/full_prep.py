import os
import numpy as np
from scipy.io import loadmat
import h5py
from scipy.ndimage.interpolation import zoom
from skimage import measure
import warnings
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from multiprocessing import Pool
from functools import partial
from step1 import step1_python, step1_python_my
import warnings
import pandas


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask

# def savenpy(id):
id = 1


def lumTrans(img):
    lungwin = np.array([-1200., 600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg*255).astype('uint8')
    return newimg

# resample norm spacing
# imgs: voxel img


def resample(imgs, spacing, new_spacing,order = 2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode='nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape) == 4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:, :, :, i]
            newslice, true_spacing = resample(slice, spacing, new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg), [1, 2, 3, 0])
        return newimg, true_spacing
    else:
        raise ValueError('wrong shape')


def savenpy(id,filelist,prep_folder,data_path,use_existing=True):      
    resolution = np.array([1, 1, 1])
    name = filelist[id]
    # print(os.path.join(prep_folder, name + '_extendbox'))
    if use_existing:
        if os.path.exists(os.path.join(prep_folder, name+'_label.npy')) and os.path.exists(os.path.join(prep_folder,name+'_clean.npy')):
            print(name+' had been done')
            return
        # if os.path.exists(os.path.join(prep_folder, name + '_extendbox.npy')):
        #     print(name+' had been done')
        #     return
    try:
        print(name)
        im, m1, m2, spacing, origin, isflip = step1_python_my(os.path.join(data_path,name))
        Mask = m1+m2
        # print('line 73')
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        xx, yy, zz = np.where(Mask)
        box = np.array([[np.min(xx), np.max(xx)], [np.min(yy), np.max(yy)], [np.min(zz), np.max(zz)]])
        box = box*np.expand_dims(spacing, 1)/np.expand_dims(resolution, 1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0]-margin], 0), np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')
        np.save(os.path.join(prep_folder, name + '_extendbox'), extendbox)
        # print('line 83')

        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170
        # print('line 93')
        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value
        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        # print('line 103')
        sliceim = sliceim2[np.newaxis, ...]
        np.save(os.path.join(prep_folder, name+'_clean'), sliceim)
        np.save(os.path.join(prep_folder, name+'_label'), np.array([[0,0,0,0]]))
    except:
        print('bug in '+name)
        raise
    print(name+' done')


def full_prep_my(step1=True,step2 = True):
    warnings.filterwarnings("ignore")
    #preprocess_result_path = './prep_result'
    prep_folder = config['preprocess_result_path']
    data_path = config['stage1_data_path']
    finished_flag = '.flag_prepkaggle'

    if True:
        alllabelfiles = config['stage1_annos_path']
        tmp = []
        for f in alllabelfiles:
            content = np.array(pandas.read_csv(f))
            content = content[content[:, 0] != np.nan]
            tmp.append(content[:, :5])
        alllabel = np.concatenate(tmp, 0)
        # filelist = os.listdir(config['stage1_data_path'])

        if not os.path.exists(prep_folder):
            os.mkdir(prep_folder)
        #eng.addpath('preprocessing/',nargout=0)

        raw_dataPath = config['stage1_data_path']

        raw_folders = [x for x in os.listdir(raw_dataPath) if
                       os.path.exists(os.path.join(raw_dataPath, x)) and not os.path.isfile(
                           os.path.join(raw_dataPath, x))]
        raw_folders = [x for x in raw_folders if x.find('train_subset') > -1 or x.find('val_subset') > -1]
        # raw_folders = [x for x in raw_folders if x.find('train_subset') > -1]

        raw_folders = np.sort(raw_folders)

        filelist = list()
        fileFolderlist = list()
        # select_file_list = ['LKDS-01000', 'LKDS-00353']

        for folder in raw_folders:
            for onefile in os.listdir(os.path.join(raw_dataPath, folder)):
                if onefile.find('.mhd') > -1 and not os.path.exists(os.path.join(prep_folder, onefile.split('.m')[0]+'_label.npy')):

                    filelist.append(onefile)
                    fileFolderlist.append(folder)

        # filelist = [f for f in os.listdir(data_path) if '.mhd' in f]
        # print(filelist)

        # for i in range(len(filelist)):
        #     savenpy_luna_my(i, annos= alllabel,filelist=filelist,fileFolderlist=fileFolderlist,data_path=data_path,savepath=prep_folder)
        print('starting preprocessing')
        pool = Pool(processes=5)

        try:
            partial_savenpy = partial(savenpy_luna_my, annos=alllabel, filelist=filelist, fileFolderlist=fileFolderlist,
                                      data_path=data_path, savepath=prep_folder)

            N = len(filelist)
            # savenpy(1)
            _ = pool.map(partial_savenpy, range(N))
            pool.close()
            pool.join()
            print('end preprocessing')
        except Exception as e:
            print(e)
    f= open(finished_flag,"w+")


def full_prep(data_path,prep_folder,n_worker = None,use_existing=True):
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

    print('starting preprocessing')
    pool = Pool(n_worker)
    # 'LKDS-00300.mhd'
    bad_file_list = ['LKDS-00385.mhd', 'LKDS-00439.mhd', 'LKDS-00322.mhd',
                     'LKDS-00818.mhd', 'LKDS-00665.mhd',
                     'LKDS-01343.mhd',
                     'LKDS-00926.mhd','LKDS-00504.mhd']
    # bad_file_list = ['LKDS-00665.mhd']
    filelist = [f for f in os.listdir(data_path) if f not in bad_file_list and 'mhd' in f]
    filelist = sorted(filelist)
    print filelist
    # filelist = filelist[:10]
    # print(filelist)
    # filelist = [f for f in os.listdir(data_path)]
    # print filelist
    # print('line 130')
    partial_savenpy = partial(savenpy, filelist=filelist, prep_folder=prep_folder,
                              data_path=data_path, use_existing=use_existing)

    N = len(filelist)
    _=pool.map(partial_savenpy, range(N))
    pool.close()
    pool.join()
    print('end preprocessing')
    return filelist

if __name__ == '__main__':
    config = {'stage1_data_path': './train_subset/',
              'preprocess_result_path': './result/',
              'stage1_annos_path': ['./annotation/annotations.csv']}
    prep_folder = config['preprocess_result_path']
    data_path = config['stage1_data_path']
    full_prep(data_path, prep_folder)

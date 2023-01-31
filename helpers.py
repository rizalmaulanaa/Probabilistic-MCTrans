import re
import numpy as np
import nibabel as nib
import tensorflow as tf
import albumentations as A
import matplotlib.pyplot as plt

from glob import glob
from keras import backend as K


def read_data(file_path, start=0, end=0, step=1):
    # Read data nii.gz 
    img = nib.load(file_path)
    data = img.get_fdata()
    end = data.shape[-1] if end == 0 else end
    reverse_channel = [data[:,:,i] for i in range(start,end,step)]
    return np.array(reverse_channel, dtype='float32')

def adjust_data(img, cf=1, mask=False):
    if mask is False:
        # Normalized data 
        img = (img - np.mean(img))/(np.std(img) + K.epsilon())

    img = np.expand_dims(img, -1)
    # Central crop if needed
    if cf<1: img = tf.image.central_crop(img, cf)
    return img

def check_data(img, seg):
    # Select data that have a value (image that contain only background)
    idx = [k for k,i in enumerate(img) if len(np.unique(i)) > 1] 
    return img[idx], seg[idx]

def data_augmentation(images, masks):
    # Data augmentation
    transform = A.Compose([A.HorizontalFlip(), A.Rotate(p=0.8)])
    for k,(i,j) in enumerate(zip(images,masks)):
        transformed = transform(image=i, mask=j)
        images[k] = transformed['image']
        masks[k] = transformed['mask']

def find_best(path_model, type_='min', model_type='det'):
    # Find best weights in filename
    if model_type == 'det':
        path = glob(path_model+'/*.hdf5') 
    else:
        path = glob(path_model+'/*') 
        
    split = [i.split('-')[-1] for i in path]
    split.sort(key=natural_keys)

    if type_ == 'min':
        file_ = split[0]
    else:
        file_ = split[-1]
    return [i for i in path if file_ in i] [0]

def show_img(img, y_true, y_pred, idx=0):
    # Visualize image, ground truth, and prediction by slice
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1); plt.title('FLAIR image')
    plt.imshow(img[idx,:,:,0], cmap='gray'); plt.axis('off')
    plt.subplot(1,3,2); plt.title('Ground Truth')
    plt.imshow(y_true[idx,:,:,0], cmap='gray'); plt.axis('off')
    plt.subplot(1,3,3); plt.title('Prediction')
    plt.imshow(y_pred[idx,:,:,0], cmap='gray'); plt.axis('off')
    plt.show()

def save_wmh(y_pred, file_in, file_out, name='ADNI'):
    # Save WMHs predictions 
    y_true = nib.load(file_in)
    wmh = np.zeros(y_true.shape)

    if name == 'Singapore':
        if wmh.shape[1] != y_pred.shape[2]:
            wmh_temp = tf.image.rot90(y_pred, k=3).numpy()
            for i in range(wmh.shape[-1]):
                wmh[:,12:220,i] = wmh_temp[i,:,:,0]
        else:
            for i in range(wmh.shape[-1]):
                wmh[12:220,:,i] = y_pred[i,:,:,0]

    elif name == 'GE3T':
        for i in range(wmh.shape[-1]):
            wmh[:128,:,i] = y_pred[i,:,:,0]
    else:
        for i in range(wmh.shape[-1]):
            wmh[:,:,i] = y_pred[i,:,:,0]

    wmh = nib.Nifti1Image(wmh, y_true.affine, y_true.header)
    nib.save(wmh, file_out)

def save_wmh_challenge(y_pred, file_in, file_out, name):
    # Save WMHs prediction only for Challenge dataset as full
    y_true = nib.load(file_in)
    wmh = np.zeros(y_true.shape)

    if name == 'Singapore': start=4; end=236
    elif name == 'GE3T': start=54; end=186
    elif name == 'Utrecht': start=8; end=248
    else: start=0; end=240

    if name == 'Singapore':
        if wmh.shape[1] != y_pred.shape[2]:
            wmh_temp = tf.image.rot90(y_pred, k=3).numpy()
            for i in range(wmh.shape[-1]):
                wmh[:,:,i] = wmh_temp[i,:,start:end,0]
        else:
            for i in range(wmh.shape[-1]):
                wmh[:,:,i] = y_pred[i,start:end,:,0]

    elif name == 'Utrecht':
        for i in range(wmh.shape[-1]):
            wmh[:,:,i] = y_pred[i,:,start:end,0]
    else:
        for i in range(wmh.shape[-1]):
            wmh[:,:,i] = y_pred[i,start:end,:,0]

    wmh = nib.Nifti1Image(wmh, y_true.affine, y_true.header)
    nib.save(wmh, file_out)

# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atoi(text):
    return int(text) if text.isdigit() else text

# https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def natural_keys(text):
    # Human sorting
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# https://github.com/baumgach/PHiSeg-code/blob/c43f3b32e1f434aecba936ff994b6f743ba7a5f8/utils.py#L326-L370
def ambiguity_map(y_gen, seg=None):
    # Claculate ambiguity maps
    def pixel_wise_xent(m_samp, m_gt, eps=1e-8):
        log_samples = np.log(m_samp + eps)
        return -1.0*np.sum(m_gt*log_samples, axis=-1)

    y_pred = np.average(y_gen, axis=0)
    E_arr = np.zeros(y_gen.shape)
    for i in range(y_gen.shape[0]):
        for j in range(y_gen.shape[1]):
            if seg is None:
                E_arr[i,j,...] = np.expand_dims(pixel_wise_xent(y_gen[i,j,...], y_pred[j,...]), axis=-1)
            else:
                E_arr[i,j,...] = np.expand_dims(pixel_wise_xent(y_gen[i,j,...], seg[j,...]), axis=-1)

    return np.average(E_arr, axis=0)

# def raw2nii(type_, train=True, bias_cor=True, path_dataset=None):
#     # Convert .raw into .nii.gz
#     train = 'training' if train else 'testing'
#     path_files = path_dataset+'NITRC/'+train+'/{}/{}'
#     path_raw = glob(path_files.format('*', '*'+type_+'.nhdr'))
#     for i in path_raw:
#         file_name = i.split('/')[-1].split('.')[0] + '.nii.gz'
#         if bias_cor:
#             bias_field_correction(i, path_files.format('full',file_name))
#         else:
#             img = sitk.ReadImage(i)
#             sitk.WriteImage(img, path_files.format('full',file_name))

# def bias_field_correction(path, out):
#     # BFC using ANTs library
#     img = nib.load(path)
#     bfc = ants.image_read(path)
#     bfc = ants.n3_bias_field_correction(bfc)
#     bfc = nib.Nifti1Image(bfc.numpy(), img.affine, img.header)
#     nib.save(bfc, out)
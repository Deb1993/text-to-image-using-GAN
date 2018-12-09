
# coding: utf-8

# In[ ]:


import h5py
import numpy as np
from PIL import Image
import io
import matplotlib as plt


# In[ ]:


datasetFile = '/datasets/ee285f-public/caltech_ucsd_birds/birds.hdf5'


# In[ ]:


split = 'test'


# In[ ]:


dataset = h5py.File(datasetFile, mode='r')
dataset_keys = [str(k) for k in dataset[split].keys()]


# In[ ]:


import os
import io
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pdb
from PIL import Image
import torch
from torch.autograd import Variable
import pdb
import torch.nn.functional as F


# In[ ]:


# def find_wrong_image(category):
#     idx = np.random.randint(len(dataset_keys))
#     example_name = dataset_keys[idx]
#     example = dataset[split][example_name]
#     _category = example['class']

#     if _category != category:
#         return example['img']

#     return find_wrong_image(category)

def find_inter_embed():
    idx = np.random.randint(len(dataset_keys))
    example_name = dataset_keys[idx]
    example = dataset[split][example_name]
    return example['embeddings']


def validate_image(img):
    img = np.array(img, dtype=float)
    if len(img.shape) < 3:
        rgb = np.empty((64, 64, 3), dtype=np.float32)
        rgb[:, :, 0] = img
        rgb[:, :, 1] = img
        rgb[:, :, 2] = img
        img = rgb

    return img.transpose(2, 0, 1)


# In[ ]:


for idx in range(len(dataset['test'])):
    example_name = dataset_keys[idx]
    example = dataset[split][example_name]

    # pdb.set_trace()
    #print "image = ", len(example)
    right_image = np.array(example['img']).tobytes()
    right_embed = np.array(example['embeddings'], dtype=float)
#    wrong_image = np.array(self.find_wrong_image(example['class'])).tobytes()
    inter_embed = np.array(find_inter_embed())

    right_image = Image.open(io.BytesIO(right_image)).resize((64, 64))
#    wrong_image = Image.open(io.BytesIO(wrong_image)).resize((64, 64))

    right_image = validate_image(right_image)
#    wrong_image = validate_image(wrong_image)

    print idx
    txt = np.array(example['txt']).astype(str)

    sample = {
            'right_images': torch.FloatTensor(right_image),
            'right_embed': torch.FloatTensor(right_embed),
#            'wrong_images': torch.FloatTensor(wrong_image),
            'inter_embed': torch.FloatTensor(inter_embed),
            'txt': str(txt)
             }

    sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
#    sample['wrong_images'] =sample['wrong_images'].sub_(127.5).div_(127.5)


# In[ ]:


example_name = dataset_keys[10874]
example = dataset[split][example_name]


# In[ ]:


str(example['txt'].value)


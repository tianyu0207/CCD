import torch.utils.data as data
import os
import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from gcloud import download_lag_unzip
from os import listdir
import itertools
from utils.utils import *
import transformations as ts
import torch.nn as nn

pos_to_diff = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, -1),
    4: (0, 1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1)
}

#

def transform_data(data, trans):
    trans_inds = np.tile(np.arange(trans.n_transforms), len(data))
    trans_data = trans.transform_batch(np.repeat(np.array(data), trans.n_transforms, axis=0), trans_inds)
    return trans_data, trans_inds

def load_trans_data(trans, dataset):
    dl = dataset
    x_train = dl.data
    x_train_trans, labels = transform_data(x_train, trans)
    x_train_trans = x_train_trans.transpose(0, 3, 1, 2)
    return x_train_trans




def get_standardized(obj):
    x = obj
    mean = x.astype(np.float32).mean(axis=0)
    return (x.astype(np.float32) - mean) / 255


def rotate_array_deterministic(data):
  """Rotate numpy array into 4 rotation angles.
  Args:
    data: data numpy array, B x H x W x C
  Returns:
    A concatenation of the original and 3 rotations.
  """
  return np.concatenate(
      [data] + [np.rot90(data, k=k, axes=(1, 2)) for k in range(1, 4)], axis=0)


def generate_coords(H, W, K):
    h = np.random.randint(0, H - K + 1)
    w = np.random.randint(0, W - K + 1)
    return h, w


def generate_coords_position(H, W, K):

    p1 = generate_coords(H, W, K)
    h1, w1 = p1

    pos = np.random.randint(8)


    J = K // 4

    K3_4 = 3 * K // 4
    h_dir, w_dir = pos_to_diff[pos]
    h_del, w_del = np.random.randint(J, size=2)

    h_diff = h_dir * (h_del + K3_4)
    w_diff = w_dir * (w_del + K3_4)

    h2 = h1 + h_diff
    w2 = w1 + w_diff

    h2 = np.clip(h2, 0, H - K)
    w2 = np.clip(w2, 0, W - K)

    p2 = (h2, w2)

    return p1, p2, pos


def task(_):
    yield

from sklearn.feature_extraction import image

def generate_coords_svdd(H, W, K):

    p1 = generate_coords(H, W, K)
    h1, w1 = p1


    J = K // 32

    h_jit, w_jit = 0, 0

    while h_jit == 0 and w_jit == 0:
        h_jit = np.random.randint(-J, J + 1)
        w_jit = np.random.randint(-J, J + 1)

    h2 = h1 + h_jit
    w2 = w1 + w_jit

    h2 = np.clip(h2, 0, H - K)
    w2 = np.clip(w2, 0, W - K)

    p2 = (h2, w2)

    return p1, p2

def extract_patch(data_tmp):
    tmp = None
    for i in range(8):
        for j in range(8):
            tmp = data_tmp[:, :, i, j, :, :] if i == 0 and j == 0 \
                else torch.cat((tmp, data_tmp[:, :, i, j, :, :]), dim=0)
    return tmp



class CutPerm(nn.Module):
    def __init__(self, max_range = 4):
        super(CutPerm, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):
        _device = input.device

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = self._cutperm(input, aug_index)

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1 - _mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = self._cutperm(input, aug_index)

        return output

    def _cutperm(self, inputs, aug_index):

        _, _, H, W = inputs.size()
        h_mid = int(H / 2)
        w_mid = int(W / 2)

        jigsaw_h = aug_index // 2
        jigsaw_v = aug_index % 2

        if jigsaw_h == 1:
            inputs = torch.cat((inputs[:, :, h_mid:, :], inputs[:, :, 0:h_mid, :]), dim=2)
        if jigsaw_v == 1:
            inputs = torch.cat((inputs[:, :, :, w_mid:], inputs[:, :, :, 0:w_mid]), dim=3)

        return inputs

shift_trans = CutPerm()


class lag_Dataset(Dataset):
    def __init__(self, root_dir=None, transform=None, train=True):
        self.transform = transform
        self.root_dir = root_dir
        self.img_folder_path = None

        self.K = 32
        self.repeat = 100
        self.data = []
        if train:
            self.img_folder_path = os.path.join(self.root_dir, 'train_set')

            self.imgs = np.asarray([f for f in listdir(self.img_folder_path)])

            for index in range(0, len(self.imgs)):
                img = io.imread(os.path.join(self.img_folder_path, self.imgs[index]))
                img = Image.fromarray(img).convert('RGB')
                img = img.resize((512, 512), Image.ANTIALIAS)
                img = img.crop((45, 45, 475, 475))
                img = img.resize((256, 256), Image.ANTIALIAS)
                img = np.array(img)
                self.data.append(img)
                self.label = torch.zeros(1)
            self.data = np.array(self.data)
            # print(self.data.shape)
            self.data = self.data.transpose((0, 3, 1, 2))  # convert to CHW

            self.data = torch.tensor(self.data)
            data_tmp = self.data.unfold(2, 32, 32).unfold(3, 32, 32)
            self.data = extract_patch(data_tmp)
            self.data = self.data.numpy()
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
            # self.label = np.vstack(self.label)


        else:
            self.nor_img_path = os.path.join(self.root_dir, 'test_set/normal_test')
            self.abn_img_path = os.path.join(self.root_dir, 'test_set/abnormal_test')
            self.nor_imgs = np.asarray([f for f in listdir(self.nor_img_path)])
            self.abn_imgs = np.asarray([f for f in listdir(self.abn_img_path)])

            for index in range(0, len(self.nor_imgs)):
                img = io.imread(os.path.join(self.nor_img_path, self.nor_imgs[index]))
                img = Image.fromarray(img).convert('RGB')

                img = np.array(img)
                self.data.append(img)
                self.label.append(0.0)
            for index in range(0, len(self.abn_imgs)):
                img = io.imread(os.path.join(self.abn_img_path, self.abn_imgs[index]))
                img = Image.fromarray(img).convert('RGB')
                img = np.array(img)
                self.data.append(img)
                # self.label.append(1.0)
                self.label = torch.zeros(1)
            self.data = np.vstack(self.data).reshape(-1, 3, 500, 500)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

            # self.label = np.vstack(self.label)

        self.n_rot = 4
        # dataset.n_rot = n_rot
        # aug_data = load_trans_data(transformer, self.data)
        # aug_data = rotate_array_deterministic(self.data)
        self.augdata = []
        self.labels = []
        # print(self.data[1].shape)
        # tmp = rotate_array_deterministic(self.data[1])
        # print(tmp.shape)
        # exit(1)
        for i in range(0, self.data.shape[0]):
            tmp = np.expand_dims(self.data[i], axis=0)
            tmp = tmp.transpose((0, 3, 1, 2))
            tmp = torch.from_numpy((tmp))

            images = torch.cat([shift_trans(tmp, k) for k in range(self.n_rot)])
            images = images.detach().cpu().numpy()
            shift_labels = torch.cat([torch.ones_like(self.label) * k for k in range(self.n_rot)], 0)  # B -> 4B
            # shift_labels = shift_labels.repeat(2)
            shift_labels = shift_labels.detach().cpu().numpy()

            self.labels.append(shift_labels)
            # print(rotate_array_deterministic(tmp).shape)
            self.augdata.append(images)
        # # print(self.augdata)
        self.data = np.concatenate(self.augdata, axis=0)
        self.data = self.data.transpose((0, 2, 3, 1))
        # print(self.data.shape)
        self.labels = np.concatenate(self.labels, axis=0)
        # print(self.labels.shape)

    def __getitem__(self, item):

        img = self.data[item]
        # patch1, patch2, pos = self.get_patch_and_pos(item)
        img_size = img.shape
        #
        # ran_p1, ran_p2 = self.get_random_patch(item)

        # img = patch1.transpose(1, 2, 0)  # convert to HWC
        # patch1 = Image.fromarray(patch1)
        # patch2 = Image.fromarray(patch1)
        # ran_p1 = Image.fromarray(ran_p1)
        # ran_p2 = Image.fromarray(ran_p2)
        img = img.astype(np.uint8)
        # print(img.shape)
        img = Image.fromarray(img)
        # img.show()
        label = self.labels[item]
        # img.show()
        # img = self.patch_transform(img)

        label = torch.as_tensor(label).long()
        # patch1, patch2, pos = self.get_patch_and_pos(item)

        # print(patch1.shape)
        # print(patch2.shape)
        # print(pos)


        if self.transform is not None:
            img = self.transform(img)
            # patch2 = self.transform(patch2)
            # ran_p1 = self.transform(patch2)
            # ran_p2 = self.transform(patch2)
        class_name = 0

        out = {'image': img, 'target': label, 'meta': {'im_size': img_size, 'index': item, 'class_name': class_name}}
        return out

    def __len__(self):
        N = len(self.data)
        return N



class lag_pos_Dataset(Dataset):
    def __init__(self, root_dir=None, transform=None, train=True):
        self.transform = transform
        self.root_dir = root_dir
        self.img_folder_path = None

        self.K = 32
        self.repeat = 100
        self.data = []
        self.label = []
        if train:
            self.img_folder_path = os.path.join(self.root_dir, 'train_set')

            self.imgs = np.asarray([f for f in listdir(self.img_folder_path)])

            for index in range(0, len(self.imgs)):
                img = io.imread(os.path.join(self.img_folder_path, self.imgs[index]))
                img = Image.fromarray(img).convert('RGB')
                img = img.resize((512, 512), Image.ANTIALIAS)
                img = img.crop((45, 45, 475, 475))
                img = img.resize((256, 256), Image.ANTIALIAS)
                img = np.array(img)
                self.data.append(img)
                self.label.append(0.0)
            self.data = np.array(self.data)

            self.label = np.vstack(self.label)

    def __getitem__(self, item):
        N = self.data.shape[0]
        K = self.K
        n = item % N
        image = self.data[n]
        image = image.transpose(2, 0, 1)   # convert to CHW

        p1, p2, pos = generate_coords_position(256, 256, K)
        patch1 = crop_image_CHW(image, p1, K).copy()
        patch2 = crop_image_CHW(image, p2, K).copy()
        patch1 = patch1.transpose(1, 2, 0)  # convert to CHW
        patch2 = patch2.transpose(1, 2, 0)  # convert to CHW

        patch1 = Image.fromarray(patch1)
        patch2 = Image.fromarray(patch2)
        if self.transform is not None:
            patch1 = self.transform(patch1)
            patch2 = self.transform(patch2)


        return patch1, patch2, pos

    def __len__(self):
        N = self.data.shape[0]
        return N * self.repeat


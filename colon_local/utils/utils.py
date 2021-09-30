"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import numpy as np
import errno
import _pickle as p
from contextlib import contextmanager
from PIL import Image
from torch.utils.data import Dataset

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        output = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)
    
    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)
    
    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' %(100*z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()





__all__ = ['crop_image_CHW', 'PatchDataset_NCHW', 'NHWC2NCHW_normalize', 'NHWC2NCHW',
           'save_binary', 'load_binary', 'makedirpath', 'task', 'DictionaryConcatDataset',
           'to_device', 'distribute_scores', 'resize']


def to_device(obj, device, non_blocking=False):
    """

    :param obj:
    :param device:
    :param bool non_blocking:
    :return:
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)

    if isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking=non_blocking)
                for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_device(v, device, non_blocking=non_blocking)
                for v in obj]

    if isinstance(obj, tuple):
        return tuple([to_device(v, device, non_blocking=non_blocking)
                     for v in obj])


@contextmanager
def task(_):
    yield


class DictionaryConcatDataset(Dataset):
    def __init__(self, d_of_datasets):
        self.d_of_datasets = d_of_datasets
        lengths = [len(d) for d in d_of_datasets.values()]
        self._length = min(lengths)
        self.keys = self.d_of_datasets.keys()
        assert min(lengths) == max(lengths), 'Length of the datasets should be the same'

    def __getitem__(self, idx):
        return {
            key: self.d_of_datasets[key][idx]
            for key in self.keys
        }

    def __len__(self):
        return self._length


def crop_CHW(image, i, j, K, S=1):
    if S == 1:
        h, w = i, j
    else:
        h = S * i
        w = S * j
    return image[:, h: h + K, w: w + K]


def cnn_output_size(H, K, S=1, P=0) -> int:
    """

    :param int H: input_size
    :param int K: filter_size
    :param int S: stride
    :param int P: padding
    :return:
    """
    return 1 + (H - K + 2 * P) // S


def crop_image_CHW(image, coord, K):
    h, w = coord
    return image[:, h: h + K, w: w + K]


class PatchDataset_NCHW(Dataset):
    def __init__(self, memmap, tfs=None, K=32, S=1):
        super().__init__()
        self.arr = memmap
        self.tfs = tfs
        self.S = S
        self.K = K
        self.N = self.arr.shape[0]

    def __len__(self):
        return self.N * self.row_num * self.col_num

    @property
    def row_num(self):
        N, C, H, W = self.arr.shape
        K = self.K
        S = self.S
        I = cnn_output_size(H, K=K, S=S)
        return I

    @property
    def col_num(self):
        N, C, H, W = self.arr.shape
        K = self.K
        S = self.S
        J = cnn_output_size(W, K=K, S=S)
        return J

    def __getitem__(self, idx):
        N = self.N
        n, i, j = np.unravel_index(idx, (N, self.row_num, self.col_num))
        K = self.K
        S = self.S
        image = self.arr[n]
        patch = crop_CHW(image, i, j, K, S)

        if self.tfs:
            patch = self.tfs(patch)

        return patch, n, i, j


def NHWC2NCHW_normalize(x):
    x = (x / 255.).astype(np.float32)
    return np.transpose(x, [0, 3, 1, 2])


def NHWC2NCHW(x):
    return np.transpose(x, [0, 3, 1, 2])


def load_binary(fpath, encoding='ASCII'):
    with open(fpath, 'rb') as f:
        return p.load(f, encoding=encoding)


def save_binary(d, fpath):
    with open(fpath, 'wb') as f:
        p.dump(d, f)


def makedirpath(fpath: str):
    dpath = os.path.dirname(fpath)
    if dpath:
        os.makedirs(dpath, exist_ok=True)


def distribute_scores(score_masks, output_shape, K: int, S: int) -> np.ndarray:
    N = score_masks.shape[0]
    results = [distribute_score(score_masks[n], output_shape, K, S) for n in range(N)]
    return np.asarray(results)


def distribute_score(score_mask, output_shape, K: int, S: int) -> np.ndarray:
    H, W = output_shape
    mask = np.zeros([H, W], dtype=np.float32)
    cnt = np.zeros([H, W], dtype=np.int32)

    I, J = score_mask.shape[:2]
    for i in range(I):
        for j in range(J):
            h, w = i * S, j * S

            mask[h: h + K, w: w + K] += score_mask[i, j]
            cnt[h: h + K, w: w + K] += 1

    cnt[cnt == 0] = 1

    return mask / cnt


def resize(image, shape=(256, 256)):
    return np.array(Image.fromarray(image).resize(shape[::-1]))

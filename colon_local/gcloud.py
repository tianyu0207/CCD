import logging
import os
import zipfile
from pathlib import Path
from typing import Union

from google.cloud import storage
from tqdm import tqdm

log = logging.getLogger(__file__)
log.setLevel(logging.DEBUG)


def get_bucket(bucket_namespace: str, bucket_name: str):
    client = storage.Client(project=bucket_namespace)
    bucket = client.get_bucket(bucket_name)
    return bucket


def download_mnist(data_dir: str):
    """Download the MNIST data from GCS to local disk with ``data_dir`` as the parent
    directory. Eg.

    data_dir/MNIST/
    └── processed
        ├── test.pt
        └── training.pt
    """
    bucket_namespace = "aiml-infrastructure"
    bucket_name = "aiml-datasets"

    bucket_prefix = "mnist/processed"
    dst_prefix = Path("MNIST") / "processed"
    dst_folder = Path(data_dir) / dst_prefix

    if dst_folder.exists():
        log.info("Skipping download as data already exists")
        return

    dst_folder.mkdir(parents=True)
    log.info(f"Downloading MNIST data from GCS to '{dst_folder}'")

    bucket = get_bucket(bucket_namespace, bucket_name)
    blobs = list(bucket.list_blobs(prefix=bucket_prefix))
    for blob in tqdm(blobs):
        stem = blob.name.split("/")[-1]  # extract last part a/b/c/filename.xyz
        dst_filepath = dst_folder / stem
        with open(dst_filepath, "wb") as sink:
            blob.download_to_file(sink)




def download_lag_unzip(data_dir: str):
    bucket_prefix = 'LAG_AD.zip'
    dst_folder = Path(data_dir)
    print('| dst folder... {}'.format(dst_folder))
    # print(os.listdir('/pvc/'))

    if dst_folder.exists():
        # print(os.listdir('/pvc/'))
        print('Skipping download as data dir already exists')
        return
    else:
        print('downloading')
        bucket = get_bucket('aiml-carneiro-research',
                            'aiml-carneiro-research-data')
        blob = bucket.blob(bucket_prefix)
        with open(Path('/pvc')/bucket_prefix, 'wb') as sink:
            blob.download_to_file(sink)
        print('unziping')
        with zipfile.ZipFile('/pvc/LAG_AD.zip', 'r') as zip_ref:
            zip_ref.extractall('/pvc/')
    print(os.listdir('/pvc/'))


def download_chestxray_unzip(data_dir: str):
    bucket_prefix = 'chestxray.zip'
    dst_folder = Path(data_dir)
    print('| dst folder... {}'.format(dst_folder))
    # print(os.listdir('/pvc/'))

    if dst_folder.exists():
        # print(os.listdir('/pvc/'))
        print('Skipping download as data dir already exists')
        return
    else:
        print('downloading')
        bucket = get_bucket('aiml-carneiro-research',
                            'aiml-carneiro-research-data')
        blob = bucket.blob(bucket_prefix)
        with open(Path('/pvc')/bucket_prefix, 'wb') as sink:
            blob.download_to_file(sink)
        print('unziping')
        with zipfile.ZipFile('/pvc/chestxray.zip', 'r') as zip_ref:
            zip_ref.extractall('/pvc/')
    print(os.listdir('/pvc/'))



def download_cifar(data_dir: str, bucket_namespace: str, bucket_name: str):
    """Download the CIFAR data from GCS to local disk with ``data_dir`` as the parent
    directory. Eg.

    data_dir/MNIST/
    └── processed
        ├── test.pt
        └── training.pt
    """
    # bucket_namespace = "aiml-infrastructure"
    # bucket_name = "aiml-datasets"

    bucket_prefix = "cifar-10-batches-py"
    # # dst_prefix = Path("MNIST") / "processed"
    # dst_prefix = Path("cifar-10-batches-py")
    # dst_folder = Path(data_dir) / dst_prefix
    dst_folder = Path(data_dir)
    print('| dst folder...')
    print(dst_folder)
    if dst_folder.exists():
        log.info("Skipping download as data already exists")
        print('| Skipping download as data already exists...')
        return

    dst_folder.mkdir(parents=True)
    log.info(f"Downloading CIFAR-10 data from GCS to '{dst_folder}'")
    print('| Downloading...')

    bucket = get_bucket(bucket_namespace, bucket_name)
    blobs = list(bucket.list_blobs(prefix=bucket_prefix))
    for blob in tqdm(blobs):
        stem = blob.name.split("/")[-1]  # extract last part a/b/c/filename.xyz
        dst_filepath = dst_folder / stem
        with open(dst_filepath, "wb") as sink:
            blob.download_to_file(sink)


def download_webvision(data_dir: str, bucket_namespace: str, bucket_name: str):
    """Download the webvision data from GCS to local disk with ``data_dir`` as the parent
    directory. Eg.

    data_dir/MNIST/
    └── processed
        ├── test.pt
        └── training.pt
    """
    # bucket_namespace = "aiml-infrastructure"
    # bucket_name = "aiml-datasets"

    bucket_prefix = "noisy-labels-datasets/data_webvision"
    # # dst_prefix = Path("MNIST") / "processed"
    # dst_prefix = Path("cifar-10-batches-py")
    # dst_folder = Path(data_dir) / dst_prefix
    dst_folder = Path(data_dir)
    print('| dst folder...')
    print(dst_folder)
    # print('printint content...')
    # print(os.listdir(dst_folder))
    print('printing blobs...')
    bucket = get_bucket(bucket_namespace, bucket_name)
    blobs = list(bucket.list_blobs(prefix=bucket_prefix))
    count = 10
    for blob in tqdm(blobs):
        print(blob)
        # extract last part a/b/c/filename.xyz
        stem = blob.name.split("data_webvision/")[-1]

        # create subfolders if exist
        parents_stem = "/".join(stem.split('/')[:-1])
        sub_folder = Path(data_dir) / parents_stem
        if sub_folder.exists() == False:
            sub_folder.mkdir(parents=True)

        dst_filepath = dst_folder / stem
        print(dst_filepath)
        with open(dst_filepath, "wb") as sink:
            blob.download_to_file(sink)
        count = count + 1
        if count > 3:
            break

    # print('printint content...')
    # print(os.listdir(dst_folder))

    if dst_folder.exists():
        log.info("Skipping download as data already exists")
        print('| Skipping download as data already exists...')
        # subfolder = Path('/pvc/data_webvision/info')
        # print(subfolder)
        # if subfolder.exist():
        #     print('subfolder exists')
        # else:
        #     print('subfolder do no exist')

        return

    dst_folder.mkdir(parents=True)
    log.info(f"Downloading CIFAR-10 data from GCS to '{dst_folder}'")
    print('| Downloading...')

    bucket = get_bucket(bucket_namespace, bucket_name)
    blobs = list(bucket.list_blobs(prefix=bucket_prefix))
    for blob in tqdm(blobs):
        stem = blob.name.split("/")[-1]  # extract last part a/b/c/filename.xyz
        dst_filepath = dst_folder / stem
        with open(dst_filepath, "wb") as sink:
            blob.download_to_file(sink)


def download_webvision_unzip(data_dir: str, bucket_namespace: str, bucket_name: str):
    """Download the webvision data from GCS to local disk with ``data_dir`` as the parent
    directory. Eg.

    data_dir/MNIST/
    └── processed
        ├── test.pt
        └── training.pt
    """
    # bucket_namespace = "aiml-infrastructure"
    # bucket_name = "aiml-datasets"

    bucket_prefix = "noisy-labels-datasets/data_webvision.zip"
    # # dst_prefix = Path("MNIST") / "processed"
    # dst_prefix = Path("cifar-10-batches-py")
    # dst_folder = Path(data_dir) / dst_prefix
    dst_folder = Path(data_dir)
    print('| dst folder...')
    print(dst_folder)

    # print('printint content...')
    # print(os.listdir(dst_folder))

    if dst_folder.exists():
        log.info("Skipping download as data already exists")
        print('| Skipping download as data already exists...')
        # subfolder = Path('/pvc/data_webvision/info')
        # print(subfolder)
        # if subfolder.exist():
        #     print('subfolder exists')
        # else:
        #     print('subfolder do no exist')

        return

    else:
        print("| Folder does not exist...")
        bucket = get_bucket(bucket_namespace, bucket_name)
        blob = bucket.blob("noisy-labels-datasets/data_webvision.zip")
        with open(Path("/pvc")/"data_webvision.zip", "wb") as sink:
            blob.download_to_file(sink)

        print('| Unziping...')
        with zipfile.ZipFile('/pvc/data_webvision.zip', 'r') as zip_ref:
            zip_ref.extractall('/pvc/')
    print(os.listdir('.'))
    print(os.listdir('/pvc/'))

    # dst_folder.mkdir(parents=True)
    # log.info(f"Downloading CIFAR-10 data from GCS to '{dst_folder}'")
    # print('| Downloading...')

    # bucket = get_bucket(bucket_namespace, bucket_name)
    # blobs = list(bucket.list_blobs(prefix=bucket_prefix))
    # for blob in tqdm(blobs):
    #     stem = blob.name.split("/")[-1]  # extract last part a/b/c/filename.xyz
    #     dst_filepath = dst_folder / stem
    #     with open(dst_filepath, "wb") as sink:
    #         blob.download_to_file(sink)


def download_clothing1M_unzip(data_dir: str, bucket_namespace: str, bucket_name: str):
    """Download the webvision data from GCS to local disk with ``data_dir`` as the parent
    directory. Eg.

    data_dir/MNIST/
    └── processed
        ├── test.pt
        └── training.pt
    """

    bucket_prefix = "noisy-labels-datasets/data_clothing.zip"

    dst_folder = Path(data_dir)
    print('| dst folder...')
    print(dst_folder)

    if dst_folder.exists():
        log.info("Skipping download as data already exists")
        print('| Skipping download as data already exists...')
        return

    else:
        print("| Folder does not exist...")
        bucket = get_bucket(bucket_namespace, bucket_name)
        blob = bucket.blob("noisy-labels-datasets/data_clothing.zip")
        with open(Path("/pvc")/"data_clothing.zip", "wb") as sink:
            blob.download_to_file(sink)

        print('| Unziping...')
        with zipfile.ZipFile('/pvc/data_clothing.zip', 'r') as zip_ref:
            zip_ref.extractall('/pvc/')
    print(os.listdir('.'))
    print(os.listdir('/pvc/'))

# def upload_checkpoint(
#     bucket_namespace: str, bucket_name: str, checkpoint_filepath: Union[Path, str]
# ):
#     """Upload a model checkpoint to the specified bucket in GCS."""
#     bucket_prefix = "filipe-research"
#     dst_path = f"{bucket_prefix}/{checkpoint_filepath}"

#     log.info(f"Uploading '{checkpoint_filepath}' => '{dst_path}'")

#     bucket = get_bucket(bucket_namespace, bucket_name)
#     blob = bucket.blob(dst_path)
#     blob.upload_from_filename(checkpoint_filepath)


def upload_checkpoint(
        bucket_namespace: str, bucket_name: str,prefix:str, checkpoint_filepath: Union[Path, str]
):
    """Upload a model checkpoint to the specified bucket in GCS."""
    bucket_prefix = prefix
    dst_path = f"{bucket_prefix}/{checkpoint_filepath}"
    # dst_path = f"{bucket_prefix}/{target_filepath}"

    print('Uploading {} => {}'.format(checkpoint_filepath,dst_path))

    bucket = get_bucket(bucket_namespace, bucket_name)
    blob = bucket.blob(dst_path)
    blob.upload_from_filename(checkpoint_filepath)


def download_checkpoint(checkpoint_filepath: str, prefix:str, bucket_namespace: str, bucket_name: str):
    src_path = f"{prefix}/{checkpoint_filepath}"
    print('Downloading {} => {}'.format(checkpoint_filepath,src_path))
    bucket = get_bucket(bucket_namespace, bucket_name)
    blob = bucket.blob(src_path)
    blob.download_to_filename(checkpoint_filepath)
    print('Finish downloading')

"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored
from models.resnet_cifar import ResNet
from gcloud import download_checkpoint, upload_checkpoint

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--class_number', default=2, type=int)
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--user', default='yu', type=str)
parser.add_argument('--dst_bucket_project',
                    default='aiml-carneiro-research', type=str)
parser.add_argument('--dst_bucket_name',
                    default='aiml-carneiro-research-data', type=str)
parser.add_argument('--upload-name', default='simclr-colon-cutout', type=str)
parser.add_argument('--gcloud', action='store_true',
                    help='use gc cloud for storing file')
# parser.add_argument('--augment-feature', action='store_true',
#                     help='use gc cloud for storing file')

parser.add_argument('--mlp_number',
                    default=2, type=int)

parser.add_argument('--cls_number',
                    default=1, type=int)
args = parser.parse_args()


def save_checkpoint(filename='checkpoint'):
    prefix = os.path.join(args.user, args.upload_name, 'checkpoints')
    upload_checkpoint(args.dst_bucket_project, args.dst_bucket_name, prefix, filename)


def main():

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))

    model = get_model(p, mlp_number=args.mlp_number, cls_head_number=args.cls_number)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)

    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)

    train_dataset = get_train_dataset(p, train_transforms, args.gcloud, to_augmented_dataset=True,
                                        split='train+unlabeled',class_num=args.class_number) # Split is for stl-10

    train_dataloader = get_train_dataloader(p, train_dataset)
    print('get train dataloader')
    print('Dataset contains {} train samples'.format(len(train_dataset)))


    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    start_epoch = 0
    model = model.cuda()

    # Training
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        simclr_train(train_dataloader, model, criterion, optimizer, epoch)


        if epoch % 50 == 0 and epoch > 200:
            filename = 'ckpt_{}.pth.tar'.format(epoch)
            torch.save(model.state_dict(), filename)
            if args.gcloud:
                save_checkpoint(filename=filename)
        # torch.save(model.state_dict(), p['pretext_model'])



if __name__ == '__main__':
    main()

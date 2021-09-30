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
from utils.train_utils import simclr_train
from termcolor import colored
from models.resnet_cifar import ResNet
from gcloud import download_checkpoint, upload_checkpoint
# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')

# DGX
parser.add_argument('--gcloud', action='store_true',
                    help='use gc cloud for storing file')
parser.add_argument('--user', default='yu', type=str)
parser.add_argument('--dst_bucket_project',
                    default='aiml-carneiro-research', type=str)
parser.add_argument('--dst_bucket_name',
                    default='aiml-carneiro-research-data', type=str)
parser.add_argument('--upload-name', default='simclr-cifar', type=str)
parser.add_argument('--class_number', default=0, type=int)
parser.add_argument('--download-name', default='simclr-download', type=str)

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

    #model = Encoder(z_dim=128)
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)

    model = model.cuda()

    #print(model.ln1.weight)


    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms, pos_transform = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    #val_transforms = get_val_transformations(p)
    #print('Validation transforms:', val_transforms)

    train_dataset, pos_Dataset = get_train_dataset(p, train_transforms, pos_transform, to_augmented_dataset=True,
                                        split='train+unlabeled', class_num=args.class_number) # Split is for stl-10
    #val_dataset = get_val_dataset(p, val_transforms)
    train_dataloader, pos_dataloader = get_train_dataloader(p, train_dataset, pos_Dataset)

    print('Dataset contains {} train samples'.format(len(train_dataset)))


    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
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
        simclr_train(train_dataloader, model, criterion, optimizer, epoch, pos_dataloader)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, p['pretext_checkpoint'])
        ckptname = 'ckpt_{}.pth.tar'.format(epoch)

        if epoch % 5 == 0 and epoch > 30:
            torch.save(model.state_dict(), ckptname)
            if args.gcloud:
                save_checkpoint(filename=ckptname)


    # Save final model
    final_ckpt_name = 'model.pth.tar'
    torch.save(model.state_dict(), final_ckpt_name)
    if args.gcloud:
        save_checkpoint(filename=final_ckpt_name)

if __name__ == '__main__':
    main()

"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
import torch.nn as nn
xent = nn.CrossEntropyLoss()
def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()

        #print(b)
        #print(c)
        #print(h)
        #print(w)
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        #print (input_.size())
        input_ = input_.cuda(non_blocking=True)
        # targets = batch['target'].cuda(non_blocking=True)
        #print(targets.size())

        output, logits = model(input_)
        output = output.view(b, 2, -1)
        labels = batch['target']

        labels = labels.cuda()
        labels = labels.repeat(2)

        loss_cls = xent(logits, labels)
        loss_simclr = criterion(output)
        loss = loss_simclr + loss_cls

        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)
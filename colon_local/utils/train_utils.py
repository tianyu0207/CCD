import torch
import torch.nn as nn
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
xent = nn.CrossEntropyLoss()

def simclr_train(train_loader, model, criterion, optimizer, epoch, pos_dataloader):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    losses_cls = []
    losses_simclr = []
    losses_total = []
    losses_pos = []
    model.train()
    pos_data_iter = iter(pos_dataloader)
    for i, batch in enumerate(train_loader):
        try:
            input1, input2, pos_gt = pos_data_iter.next()
        except:
            pos_data_iter = iter(pos_dataloader)
            input1, input2, pos_gt = pos_data_iter.next()

        input1, input2 = input1.cuda(non_blocking=True), input2.cuda(non_blocking=True)
        pos_gt = pos_gt.cuda(non_blocking=True)

        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()

        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)
        #print (input_.size())
        input_ = input_.cuda(non_blocking=True)
        labels = batch['target'].cuda(non_blocking=True)
        #print(targets.size())

        output, logits, pred_pos = model(input_, input1, input2)
        loss_pos = xent(pred_pos, pos_gt)
        output = output.view(b, 2, -1)
        labels = labels.repeat(2)
        labels = labels.cuda(non_blocking=True)

        loss_cls = xent(logits, labels)

        loss = criterion(output)
        total_loss = loss + loss_cls + loss_pos

        losses_cls.append(loss_cls.detach().cpu().numpy())
        losses_simclr.append(loss.detach().cpu().numpy())
        losses_total.append(total_loss.detach().cpu().numpy())

        losses.update(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)
            loss_mean_simclr = np.array(losses_simclr).mean()
            loss_mean_clr = np.array(losses_cls).mean()
            loss_mean_total = np.array(losses_total).mean()
            loss_mean_pos = np.array(losses_pos).mean()

            writer.add_scalar('Loss/cls', loss_mean_clr, epoch)
            writer.add_scalar('Loss/simclr', loss_mean_simclr, epoch)
            writer.add_scalar('Loss/pos', loss_mean_pos, epoch)
            writer.add_scalar('Loss/total', loss_mean_total, epoch)





import argparse
import os
import shutil
import time
import sys

import numpy as np
np.random.seed(0)
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
#import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models as models

import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.ndimage import binary_fill_holes

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main(argv, best_prec1=0):
    args = get_args(argv)

    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    #os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=3, num_channels=1)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr
                                ,momentum=args.momentum,
                                weight_decay=args.weight_decay
                                   )
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    try:
        if args.evaluate == False:
            train_data_x, good_ix_train = load_pickle(args.train_x_pkl, isx=True, good_ix=None)
            train_data_y,_ = load_pickle(args.train_y_pkl, isx=False, good_ix=good_ix_train)
            train_data_ids,_ = load_pickle(args.train_id_pkl, isx=False, good_ix=good_ix_train, isIDs=True)
            print('=> loading data done. train shape:', train_data_x.shape)
            
            valid_data_x, good_ix_valid = load_pickle(args.valid_x_pkl, isx=True, good_ix=None)
            valid_data_y,_ = load_pickle(args.valid_y_pkl, isx=False, good_ix=good_ix_valid)
            valid_data_ids,_ = load_pickle(args.valid_id_pkl, isx=False, good_ix=good_ix_valid, isIDs=True)
            print('=> loading data done. valid shape:', train_data_x.shape)
        else:
            test_data_x, good_ix_test = load_pickle(args.test_x_pkl, isx=True, good_ix=None)
            test_data_y,_ = load_pickle(args.test_y_pkl, isx=False, good_ix=good_ix_test)
            test_data_ids,_ = load_pickle(args.test_id_pkl, isx=False, good_ix=good_ix_test, isIDs=True)
            print('=> loading data done. test shape:', train_data_x.shape)
        
        if False: #i.e. right now all models supposedly handle single channel but to be checked later.
            train_data_x = make_3_channel(train_data_x)
            valid_data_x = make_3_channel(valid_data_x)
    except:
        print('problem loading pickles')
        raise
        
    transform = None
    batch_size=args.batch_size

    if args.evaluate:
        validate(test_data_x, test_data_y, 0 , model, criterion, args, True)
        return

    train_loss_list = []
    valid_loss_list = []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        
        doplot = True if ((epoch %10 == 0) and (epoch > 0)) else False

        # train for one epoch
        trainloss = train(train_data_x, train_data_y, train_data_ids , model, criterion, 
              optimizer, epoch, transform, batch_size, args, doplot)
        train_loss_list.append(trainloss)

        # evaluate on validation set
        prec1 = validate(valid_data_x, valid_data_y, valid_data_ids , model, 
                         criterion, batch_size, args, doplot)

        valid_loss_list.append(prec1)
        plt.figure()
        plt.plot(valid_loss_list, label='validation loss')
        plt.plot(train_loss_list, label='train loss')
        plt.legend()
        plt.show()
        
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_fname)

def train(x, y, ids, model, criterion, optimizer, epoch, transform, batch_size, args, doplot):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    preds = np.zeros(len(y))
    outs = np.zeros((len(y), 3))
    
    #reshuffle the training data 
    ix_shuffle = list(np.arange(0, len(x)))
    np.random.shuffle(ix_shuffle)
    # switch to train mode
    model.train()

    end = time.time()
    for bix in range(int(len(ix_shuffle)/batch_size)):
        i = ix_shuffle[bix*batch_size:(bix+1)*batch_size]
        input = torch.from_numpy(x[i,:,:,:]).float()
        target = torch.from_numpy(y[i]).type(torch.LongTensor)
        
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        _, preds_i = (output.data.topk(1, 1, True, True))
        outs[i,:] = output.data.cpu().numpy()
        preds[i] = preds_i.cpu().numpy()
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        accuracies.update(acc[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if bix % (int(args.print_freq/batch_size)) == 0:
            print('   Epoch: [{0}][{1}/{2}]  '
                  #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  #'Data {data_time.val:.3f} ({data_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Acc {accu.val:.3f} ({accu.avg:.3f})'.format(
                   epoch, bix, (len(x)/batch_size), batch_time=batch_time,
                   data_time=data_time, loss=losses, accu=accuracies))
    acc, auc, acc_list, auc_list = numpy_evaluate(x, preds, y, outs, ids, doplot, 'train')
    print('*   Train acc:{0:.3f}, acc_C0:{2:.3f}, acc_C1:{4:.3f}, acc_C2:{6:.3f}'.format(acc, auc, acc_list[0], auc_list[0], acc_list[1], auc_list[1], acc_list[2], auc_list[2]))
    return losses.avg


def validate(xnumpy, ynumpy, ids, model, criterion, batch_size, args, doplot):
    #y = torch.from_numpy(ynumpy).type(torch.LongTensor)
    #x = torch.from_numpy(xnumpy).float()
    preds = np.zeros(len(ynumpy))
    outs = np.zeros((len(ynumpy), 3))

    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
    ix_all = range(0,len(ynumpy))
    for bix in range(int(len(ynumpy)/batch_size)):
        i = ix_all[bix*batch_size:(bix+1)*batch_size]
        input = torch.from_numpy(xnumpy[i,:,:,:]).float()
        target = torch.from_numpy(ynumpy[i]).type(torch.LongTensor)
        #input = x[i:i+1]
        #target = y[i:i+1]
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        _, preds_i = output.data.topk(1, 1, True, True)
        outs[i,:] = output.data.cpu().numpy()
        preds[i] = preds_i.cpu().numpy()
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        acc = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        accuracies.update(acc[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if bix % args.print_freq == 0:
            print('   Epoch-Valid: [{0}/{1}]  '
                  #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Acc {accu.val:.3f} ({accu.avg:.3f})'.format(
                   bix, (len(xnumpy)/batch_size), batch_time=batch_time, loss=losses,
                   accu=accuracies))

    acc, auc, acc_list, auc_list = numpy_evaluate(xnumpy, preds, ynumpy, outs, ids, doplot, 'validation')
    print('*   Valid acc:{0:.3f}, acc_C0:{2:.3f}, acc_C1:{4:.3f}, acc_C2:{6:.3f}'.format(acc, auc, acc_list[0], auc_list[0], acc_list[1], auc_list[1], acc_list[2], auc_list[2]))
    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def aggregate_score(x, pred, y, outs, ids, doplot=False, tag=''):
    pass
        
def numpy_evaluate(x, pred, y, outs, ids, doplot=False, tag=''):
    if doplot == True:
        for ix in range(len(y)):
            print('    ',ix, ids[ix], y[ix], pred[ix], outs[ix], '(', tag, ')' )
    aggregate_score(x, pred, y, outs, ids, doplot=False, tag='')
    prec_list = [0]*int(max(y)+1)
    auc_list = [0]*int(max(y)+1)
    random_subset = np.arange(len(y))
    #np.random.shuffle(random_subset)
    visualize=200
    if doplot==True:
        f, axarray = plt.subplots(20,10, figsize=(17,30))
        for i in range(axarray.shape[0]):
            for j in range(axarray.shape[1]):
                ix_orig = random_subset[i*10+j]
                axarray[i,j].imshow(x[ix_orig].reshape(256,256))
                axarray[i,j].set_title('{0}/{1}:{2}'.format(y[ix_orig], pred[ix_orig], ids[ix_orig]), size=8)
        plt.show()
    for class_id in range(max(y)+1):
        print('    ----------------------------------     ')
        print('    |class =', class_id)
        class_ix = (y == class_id)
        class_pred_ix = (pred == class_id)
        subset_pred = pred[class_ix]
        print('    |total true pos:', class_ix.sum())
        print('    |of those, model detected:', (subset_pred==class_id).sum())
        print('    |total predicted pos:', class_pred_ix.sum())
        print('    |recall:', (subset_pred==class_id).sum()/class_ix.sum())
        print('    |precision:', (subset_pred==class_id).sum()/class_pred_ix.sum())
        prec_list[class_id] = (subset_pred==class_id).sum()/class_pred_ix.sum()
    print('    ----------------------------------     ')        
    return metrics.accuracy_score(pred, y), 0, prec_list, auc_list
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True) #argmax
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    correct_k = correct.view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
    return res[0]

def make_3_channel(raw):
    return np.stack([raw, raw, raw], axis=1)
    
def load_pickle(fname, isx=False, good_ix=0, isIDs=False):
    raw = pickle.load(open(fname, 'rb'))
    if isIDs == True:
        return np.array(raw)[good_ix], -1
    if 'T1' in fname:
        threshold = 1.5
    if 'T2' in fname:
        threshold = 1.7
    if 'GD' in fname:
        threshold = 2
    if isx:
        print('=> X data is of range', raw.min(), raw.max(), ' and of shape:', raw.shape)
        print('=> normalizing X')
    else:
        print('=> Y data is of range', raw.min(), raw.max(), ' and of shape:', raw.shape)
        print('=> Y class distribution is:', raw.sum(axis=0), ' of total', raw.shape)
    
    if isx == True:       
        #m1 = raw.mean(axis=1).ravel(); ix_m = (m1 < m1.mean() - threshold * m1.std()); print(ix_m.sum());
        #s1 = raw.std(axis=1).ravel(); ix_s = (s1 < s1.mean() - 2 * s1.std()); print(ix_s.sum());
        #both_ix = (ix_m & ix_s)
        #good_ix = (both_ix == False)
        #print('of total of', good_ix.shape, 'keeping:', good_ix.sum())
        #raw = raw[good_ix, :]        
        rawnnz = (raw > raw.mean())
        rawnnz = binary_fill_holes(rawnnz)
        good_ix = (rawnnz.sum(axis=1) > (256*256/4)) & (rawnnz.sum(axis=1) < (256*256/2)) 
        mean_all = raw[rawnnz].mean()
        std_all = raw[rawnnz].std()
        #raw = raw/255.0
        raw = rawnnz * (raw - mean_all)/(std_all) # i.e. normalize nonzeros only
        raw = raw.reshape(raw.shape[0], 1, 256, 256)
        raw = raw[good_ix, :, :]
        return raw, good_ix
    else:
        raw = np.argmax(raw, axis=1)
        raw = raw[good_ix]
        return raw, -1

def get_args(argv):
    parser = argparse.ArgumentParser(description='PyTorch Convnet Training')
    parser.add_argument('--train_x_pkl',
                        help='path to dataset pickle file train x')
    parser.add_argument('--valid_x_pkl', 
                        help='path to dataset pickle file valid x')
    parser.add_argument('--train_y_pkl',
                        help='path to dataset pickle file train y ')
    parser.add_argument('--valid_y_pkl',
                        help='path to dataset pickle file valid y')
    parser.add_argument('--train_id_pkl', help='path to dataset pickle file train ID ')
    parser.add_argument('--valid_id_pkl', help='path to dataset pickle file valid ID')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning_rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=1000, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', 
                        default=False, help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true'
                        , default=False, help='use pre-trained model')
    parser.add_argument('--gpu_id', default='2', 
                        help='GPU id to be used')
    parser.add_argument('--save_fname', default='./checkpoint.pth.tar',
                        help='name of the checkpoint file')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(sys.argv[1:])



"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import provider
import importlib
import shutil
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size in training [default: 2]')
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes [default: 2]')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--data_path', default='./data/modelnet6_ply/', help='path to dataset root file [default: ./data/modelnet6_ply/]')
    parser.add_argument('--data_extension', default='.npy', help='extension of data files [default: .npy')
    parser.add_argument('--split_name', default='modelnet6', help='split files used from dataset folder [default: modelnet6')
    parser.add_argument('--experiment_dir', default='./log/', help='log dir output [default: ./log/')
    parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--no_gpu', type=bool, default=False, help='set to true if no gpu run is required')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--uniform', type=bool, default=False, help='set to true if uniform distribution should be used')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()

def test(model, loader, criterion, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class,3))
    tot_loss = 0
    batch_tqdm = tqdm(enumerate(loader), total=len(loader))
    for j, data in batch_tqdm:
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        if not args.no_gpu:
            points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        pred,trans_feat = classifier(points)
        loss = criterion(pred, target.long(), trans_feat)
        tot_loss += loss
        pred_choice = pred.data.max(1)[1]
        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
            class_acc[cat,0]+= classacc.item()/float(points[target==cat].size()[0])
            class_acc[cat,1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item()/float(points.size()[0]))
        mean_loss = tot_loss/(j+1)
        batch_tqdm.set_description(f"loss {mean_loss}, batch ({j}/{len(loader)})")
    class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
    class_acc = np.mean(class_acc[:,2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc, mean_loss


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    if not args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path(args.experiment_dir)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('classification')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    tensorboard_dir = experiment_dir.joinpath('tensorboard/')
    tensorboard_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''TENSORBOARD'''
    train_writer = SummaryWriter(tensorboard_dir.joinpath("train"))
    val_writer = SummaryWriter(tensorboard_dir.joinpath("validation"))

    '''DATA LOADING'''
    log_string('Load dataset ...')
    DATA_PATH = args.data_path

    class_in_filename = False if args.data_extension == ".npy" else True

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, split_name=args.split_name, extension=args.data_extension, npoint=args.num_point, split='train',
                                                     normal_channel=args.normal, class_in_filename=class_in_filename, uniform=args.uniform)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, split_name=args.split_name, extension=args.data_extension, npoint=args.num_point, split='validation',
                                                    normal_channel=args.normal, class_in_filename=class_in_filename, uniform=args.uniform)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=0)

    '''MODEL LOADING'''
    num_class = args.num_classes
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('./models/pointnet_util.py', str(experiment_dir))

    classifier = MODEL.get_model(num_class, normal_channel=args.normal)
    if not args.no_gpu:
        classifier = classifier.cuda()
    criterion = MODEL.get_loss()
    if not args.no_gpu:
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50, min_lr=0.000001)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    mean_correct = []

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        mean_correct = []
        batch_tqdm = tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9)
        total_loss = 0
        predictions_likelihood_tot = torch.zeros([len(trainDataLoader.dataset), num_class])

        for batch_id, data in batch_tqdm:
            points, target = data
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.random_scale_point_cloud(points[:,:, 0:3])
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            if not args.no_gpu:
                points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
            total_loss += loss
            mean_loss = total_loss / (batch_id + 1)
            batch_tqdm.set_description(f"loss {mean_loss}, batch ({batch_id}/{len(trainDataLoader)})")
            preds_likelihood = torch.exp(pred)
            predictions_likelihood_tot[epoch*trainDataLoader.batch_size:(batch_id+1)*trainDataLoader.batch_size] = preds_likelihood

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        train_writer.add_scalar('Loss', mean_loss, epoch)
        train_writer.add_scalar('Accuracy', train_instance_acc, epoch)
        for cls in range(num_class):
            train_writer.add_histogram(f"class_{cls}", predictions_likelihood_tot[:, cls], epoch)

        with torch.no_grad():
            instance_acc, class_acc, val_loss = test(classifier.eval(), testDataLoader, criterion, num_class=num_class)
            scheduler.step(val_loss)
            val_writer.add_scalar('Loss', val_loss, epoch)
            val_writer.add_scalar('Accuracy', instance_acc, epoch)
            val_writer.add_scalar('Class_Accuracy', class_acc, epoch)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)

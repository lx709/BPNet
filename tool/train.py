import os
import time
import random
import numpy as np
import logging
import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
from os.path import join
import torch.nn.functional as F
from MinkowskiEngine import SparseTensor, CoordsManager
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU, poly_learning_rate, save_checkpoint

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
best_iou = 0.0

import pudb

def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def get_parser():
    parser = argparse.ArgumentParser(description='BPNet')
    parser.add_argument('--config', type=str, default='config/compat/bpnet_10.yaml', help='config file')
    parser.add_argument('opts', help='see config/compat/bpnet_10.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = True
        cudnn.deterministic = True
    # Log for check version
    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_iou
    args = argss

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = get_model(args)
    if args.sync_bn_2d:
        print("using DDP synced BN for 2D")
        try:
            model.layer0_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.layer0_2d)
            model.layer1_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.layer1_2d)
            model.layer2_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.layer2_2d)
            model.layer3_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.layer3_2d)
            model.layer4_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.layer4_2d)
            model.up4_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.up4_2d)
            model.delayer4_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.delayer4_2d)
            model.up3_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.up3_2d)
            model.delayer3_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.delayer3_2d)
            model.up2_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.up2_2d)
            model.delayer2_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.delayer2_2d)
            model.cls_2d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.cls_2d)
        except:
            print("3d modeling")

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Num. Part Classes: {}".format(args.classes))
        logger.info("Num. Material Classes: {}".format(args.mat))
        logger.info("Num. Objct categories: {}".format(args.categories))
        logger.info("Use separate 2d classifier: {}".format(args.use_2d_classifier))
        logger.info("Use separate 3d classifier: {}".format(args.use_3d_classifier))
        logger.info(model)

    # ####################### Optimizer ####################### #
    if args.arch == 'bpnet':
        modules_ori = [model.layer0_2d, model.layer1_2d, model.layer2_2d, model.layer3_2d, model.layer4_2d]
        modules_new = [
            model.up4_2d, model.delayer4_2d, model.up3_2d, model.delayer3_2d, model.up2_2d, model.delayer2_2d,
            model.cls_2d, model.cls_mat,
            model.layer0_3d, model.layer1_3d, model.layer2_3d, model.layer3_3d, model.layer4_3d, model.layer5_3d,
            model.layer6_3d, model.layer7_3d, model.layer8_3d, model.layer9_3d, model.cls_3d, model.cls_3dmat,
            model.linker_p2, model.linker_p3, model.linker_p4, model.linker_p5, 
        ]
        if args.use_2d_classifier:
            modules_ori.append(model.fc) # need to use a small lr for classification
        if args.use_3d_classifier:
            modules_ori.append(model.fc3d) # need to use a small lr for classification
        
        params_list = []
        for module in modules_ori:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr))
        for module in modules_new:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
        args.index_split = len(modules_ori)
        
        optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)
    else:
        model = model.cuda()
#         model = torch.nn.DataParallel(model).cuda() # this will split batch into different gpus and will cause error

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda(gpu)

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # ####################### Data Loader ####################### #

    if args.data_name == '3dcompat':
        
        import webdataset as wds
        from dataset.GCR_loader import GCRLoader3D

        ROOT_URL = args.data_root

        train_loader = GCRLoader3D(root_url=ROOT_URL, split='train', n_comp=args.com, view_type=args.view_type, sem_level=args.sem_level).make_loader(batch_size=args.batch_size, num_workers=args.workers, aug=args.aug, voxelSize=args.voxelSize, distributed=args.distributed, world_size=args.world_size)

        if args.evaluate:
            val_loader = GCRLoader3D(root_url=ROOT_URL, split='valid', n_comp=args.com, view_type=args.view_type, sem_level=args.sem_level).make_loader(batch_size=args.batch_size_val, num_workers=args.workers, aug=args.aug, voxelSize=args.voxelSize, distributed=args.distributed, world_size=args.world_size)
                
    else:
        raise Exception('Dataset not supported yet'.format(args.data_name))

    # ####################### Train ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            if args.evaluate:
                sampler = getattr(val_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

        if args.data_name == '3dcompat':
            loss_train, mIoU_train, mAcc_train, allAcc_train, \
            loss_train_2d, mIoU_train_2d, mAcc_train_2d, allAcc_train_2d \
                = train_cross(train_loader, model, criterion, optimizer, epoch)
        else:
            pass
            
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)
            if args.data_name == '3dcompat':
                writer.add_scalar('loss_train_2d', loss_train_2d, epoch_log)
                writer.add_scalar('mIoU_train_2d', mIoU_train_2d, epoch_log)
                writer.add_scalar('mAcc_train_2d', mAcc_train_2d, epoch_log)
                writer.add_scalar('allAcc_train_2d', allAcc_train_2d, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            if args.data_name == '3dcompat':
                loss_val, mIoU_val, mAcc_val, allAcc_val, \
                loss_val_2d, mIoU_val_2d, mAcc_val_2d, allAcc_val_2d \
                    = validate_cross(val_loader, model, criterion)
            else:
                loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                if args.data_name == '3dcompat':
                    writer.add_scalar('loss_val_2d', loss_val_2d, epoch_log)
                    writer.add_scalar('mIoU_val_2d', mIoU_val_2d, epoch_log)
                    writer.add_scalar('mAcc_val_2d', mAcc_val_2d, epoch_log)
                    writer.add_scalar('allAcc_val_2d', allAcc_val_2d, epoch_log)
                # remember best iou and save checkpoint
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_iou': best_iou
                }, is_best, os.path.join(args.save_path, 'model')
            )
    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def get_model(cfg):
    if cfg.arch == 'mink_18A':
        from models.unet_3d import MinkUNet18A as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3)
    elif cfg.arch == 'mink_34C':
        from models.unet_3d import MinkUNet34C as Model
        model = Model(in_channels=3, out_channels=cfg.classes, D=3)
    if cfg.arch == 'bpnet':
        from models.bpnet import BPNet as Model
        model = Model(cfg=cfg)
    else:
        raise Exception('architecture not supported yet'.format(cfg.arch))
    return model


def train_cross(train_loader, model, criterion, optimizer, epoch):
    # raise NotImplemented
    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meter, loss_meter_3d, loss_meter_2d, loss_meter_cls = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
    target_meter_mat, union_meter_mat, intersection_meter_mat = AverageMeter(), AverageMeter(), AverageMeter()
    target_meter_3dmat, union_meter_3dmat, intersection_meter_3dmat = AverageMeter(), AverageMeter(), AverageMeter()
    loss_meter_mat, loss_meter_3dmat = AverageMeter(), AverageMeter()
    acc = 0
    total = 0
    model.train()
    end = time.time()
#     max_iter = args.epochs * len(train_loader)
    max_iter = args.epochs * train_loader.length
    
    for i, batch_data in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        if args.data_name == '3dcompat':
            (coords, feat, label_3d, color, label_2d, link, cls, mat, mat_3d, model_id, _, _, _) = batch_data
            # For some networks, making the network invariant to even, odd coords is important
            coords[:, 1:4] += (torch.rand(3) * 100).type_as(coords)
            
            sinput = SparseTensor(feat.cuda(non_blocking=True), coords) #allow_duplicate_coords=True
            color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)

            label_3d, label_2d = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)
            cls, mat, mat_3d = cls.cuda(non_blocking=True), mat.cuda(non_blocking=True), mat_3d.cuda(non_blocking=True)

#             print('model inputs:', sinput.shape, color.shape, link.shape, cls.shape)
            output_3d, output_2d, output_mat, output_3dmat, output_cls = model(sinput, color, link)
            
#             print('model outputs:', output_3d.shape, output_2d.shape, output_mat.shape, output_3dmat.shape, output_cls.shape)
            
            loss_2d = criterion(output_2d, label_2d)
            loss_mat = criterion(output_mat, mat)
        
            loss_3d = criterion(output_3d, label_3d)
            loss_3dmat = criterion(output_3dmat, mat_3d)
            
#             print('output_cls, cls.max', output_cls.shape, output_cls[0], cls)
            assert cls.max() <= args.categories - 1
            loss_cls = criterion(output_cls, cls)
            
            if args.use_3d_classifier:
                loss = loss_3d + loss_3dmat + args.weight_2d * (loss_2d + loss_mat) + loss_cls
            else:
                loss = loss_3d + loss_3dmat + args.weight_2d * (loss_2d + loss_mat + loss_cls)
        else:
            raise NotImplemented
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ############ 3D ############ #
        output_3d = output_3d.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes,
                                                              args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_3d.update(intersection)
        union_meter_3d.update(union)
        target_meter_3d.update(target)
        accuracy_3d = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)

        # ############ mat_3d ############ #
        output_3dmat = output_3dmat.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_3dmat, mat_3d.detach(), args.mat,
                                                              args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_3dmat.update(intersection)
        union_meter_3dmat.update(union)
        target_meter_3dmat.update(target)
        accuracy_3dmat = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)

        # ############ 2D ############ #
        output_2d = output_2d.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_2d, label_2d.detach(), args.classes,
                                                              args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_2d.update(intersection)
        union_meter_2d.update(union)
        target_meter_2d.update(target)
        accuracy_2d = sum(intersection_meter_2d.val) / (sum(target_meter_2d.val) + 1e-10)
        
        # ############ cls ############ #
        output_cls = output_cls.detach().max(1)[1]
        correct_guessed = output_cls == cls
        cls_b_acc = torch.sum(correct_guessed.double()).item()
        acc += cls_b_acc
        total += output_cls.size(0)
        
        # ############ materials ############ #
        output_mat = output_mat.detach().max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output_mat, mat.detach(), args.mat,
                                                              args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter_mat.update(intersection)
        union_meter_mat.update(union)
        target_meter_mat.update(target)
        accuracy_mat = sum(intersection_meter_mat.val) / (sum(target_meter_mat.val) + 1e-10)

        loss_meter.update(loss.item(), args.batch_size)
        loss_meter_2d.update(loss_2d.item(), args.batch_size)
        loss_meter_3d.update(loss_3d.item(), args.batch_size)
        loss_meter_cls.update(loss_cls.item(), args.batch_size)
        loss_meter_mat.update(loss_mat.item(), args.batch_size)
        loss_meter_3dmat.update(loss_3dmat.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        # 2d compat result:

        # 3d compat resutl: Todo
        # Adjust lr
        current_iter = epoch * train_loader.length + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
        # if args.arch == 'cross_p5' or args.arch == 'cross_p2':
        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        # else:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = current_lr

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}'
                        'Loss {loss_meter_2d.val:.4f}'
                        'Accuracy2D {accuracy_2d:.4f}'
                        'Acc Cls {acc:.4f}'
                        'Loss_mat {loss_meter_mat.val:.4f} '
                        'Materials ACC {acc_mat:.4f}. '.format(epoch + 1, args.epochs, i + 1, train_loader.length,
                                                               batch_time=batch_time, data_time=data_time,
                                                               remain_time=remain_time,
                                                               loss_meter=loss_meter_3d,
                                                               accuracy=accuracy_3d,
                                                               loss_meter_2d=loss_meter_2d,
                                                               accuracy_2d=accuracy_2d,
                                                               acc=acc / total, loss_meter_mat=loss_meter_mat,
                                                               acc_mat=accuracy_mat))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('loss3d_train_batch', loss_meter_3d.val, current_iter)
            writer.add_scalar('loss2d_train_batch', loss_meter_2d.val, current_iter)
            writer.add_scalar('mIoU3d_train_batch', np.mean(intersection_meter_3d.val / (union_meter_3d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('mAcc3d_train_batch', np.mean(intersection_meter_3d.val / (target_meter_3d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('allAcc3d_train_batch', accuracy_3d, current_iter)

            writer.add_scalar('mIoU2d_train_batch', np.mean(intersection_meter_2d.val / (union_meter_2d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('mAcc2d_train_batch', np.mean(intersection_meter_2d.val / (target_meter_2d.val + 1e-10)),
                              current_iter)
            writer.add_scalar('allAcc2d_train_batch', accuracy_2d, current_iter)

            writer.add_scalar('learning_rate', current_lr, current_iter)

    iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
    accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
    mIoU_3d = np.mean(iou_class_3d)
    mAcc_3d = np.mean(accuracy_class_3d)
    allAcc_3d = sum(intersection_meter_3d.sum) / (sum(target_meter_3d.sum) + 1e-10)

    iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
    accuracy_class_2d = intersection_meter_2d.sum / (target_meter_2d.sum + 1e-10)
    mIoU_2d = np.mean(iou_class_2d)
    mAcc_2d = np.mean(accuracy_class_2d)
    allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)

    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs,
                                                                                           mIoU_3d, mAcc_3d, allAcc_3d))
        logger.info(
            'Train result 2d at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch + 1, args.epochs,
                                                                                              mIoU_2d, mAcc_2d,
                                                                                              allAcc_2d))
        logger.info('Train result cls at at epoch [{}/{}]: acc:{:.4f}/total:{}'.format(epoch + 1, args.epochs,
                                                                                       acc / total, total))

    return loss_meter_3d.avg, mIoU_3d, mAcc_3d, allAcc_3d, \
           loss_meter_2d.avg, mIoU_2d, mAcc_2d, allAcc_2d


def validate_cross(val_loader, model, criterion):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    loss_meter, loss_meter_3d, loss_meter_2d, loss_meter_cls = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
    loss_meter_mat = AverageMeter()
    target_meter_3dmat, union_meter_3dmat, intersection_meter_3dmat = AverageMeter(), AverageMeter(), AverageMeter()
    loss_meter_3dmat = AverageMeter()
    target_meter_mat, union_meter_mat, intersection_meter_mat = AverageMeter(), AverageMeter(), AverageMeter()
    acc = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        # outseg, outcls, outmat, gtcls, gtseg, gtmat, gtseg3d, outseg3d = [], [], [], [], [], [], [], []
        for i, batch_data in enumerate(val_loader):
            if args.data_name == '3dcompat':
                (coords, feat, label_3d, color, label_2d, link, cls, mat, mat3d, model_id, inds_reverse, _, _) = batch_data
                sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
                color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
                label_3d, label_2d, = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)
                cls, mat = cls.cuda(non_blocking=True), mat.cuda(non_blocking=True)
                mat3d = mat3d.cuda(non_blocking=True)
                output_3d, output_2d, output_mat, output_3dmat, output_cls = model(sinput, color, link)
                
#                 print('output_3d, output_3dmat', output_3d.shape, output_3dmat.shape, inds_reverse.max())
                output_3d = output_3d[inds_reverse, :]
                output_3dmat = output_3dmat[inds_reverse, :]

            else:
                raise NotImplemented
            
#             print('model outputs:', output_3d.shape, output_2d.shape, output_mat.shape, output_3dmat.shape, output_cls.shape)
#             print('label:', label_3d.shape, label_2d.shape, mat.shape, mat3d.shape, cls.shape)
            
            loss_2d = criterion(output_2d, label_2d)
            loss_mat = criterion(output_mat, mat)
            
            loss_3d = criterion(output_3d, label_3d)
            loss_3dmat = criterion(output_3dmat, mat3d)
            
            loss_cls = criterion(output_cls, cls)

            loss = loss_3d + loss_3dmat + args.weight_2d * (loss_2d * loss_mat + loss_cls)

            # ############ 3D ############ #
            output_3d = output_3d.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3d.update(intersection)
            union_meter_3d.update(union)
            target_meter_3d.update(target)
            accuracy_3d = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)

            # ############ 2D ############ #
            output_2d = output_2d.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_2d, label_2d.detach(), args.classes,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_2d.update(intersection)
            union_meter_2d.update(union)
            target_meter_2d.update(target)
            accuracy_2d = sum(intersection_meter_2d.val) / (sum(target_meter_2d.val) + 1e-10)
            # ############ mat_3d ############ #
            output_3dmat = output_3dmat.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3dmat, mat3d.detach(), args.mat,
                                                                  args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3dmat.update(intersection)
            union_meter_3dmat.update(union)
            target_meter_3dmat.update(target)
            accuracy_3dmat = sum(intersection_meter_3dmat.val) / (sum(target_meter_3dmat.val) + 1e-10)

            # ############ mat ############ #
            output_mat = output_mat.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_mat, mat.detach(), args.mat,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_mat.update(intersection)
            union_meter_mat.update(union)
            target_meter_mat.update(target)
            accuracy_mat = sum(intersection_meter_mat.val) / (sum(target_meter_mat.val) + 1e-10)

            loss_meter.update(loss.item(), args.batch_size)
            loss_meter_cls.update(loss_cls.item(), args.batch_size)
            loss_meter_2d.update(loss_2d.item(), args.batch_size)
            loss_meter_3d.update(loss_3d.item(), args.batch_size)
            loss_meter_mat.update(loss_mat.item(), args.batch_size)
            loss_meter_3dmat.update(loss_3dmat.item(), args.batch_size)
            # ############ cls ############ #
            output_cls = output_cls.detach().max(1)[1]
            correct_guessed = output_cls == cls
            cls_b_acc = torch.sum(correct_guessed.double()).item()
            acc += cls_b_acc
            total += output_cls.size(0)

    iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
    accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
    mIoU_3d = np.mean(iou_class_3d)
    mAcc_3d = np.mean(accuracy_class_3d)
    allAcc_3d = sum(intersection_meter_3d.sum) / (sum(target_meter_3d.sum) + 1e-10)

    iou_class_2d = intersection_meter_2d.sum / (union_meter_2d.sum + 1e-10)
    accuracy_class_2d = intersection_meter_2d.sum / (target_meter_2d.sum + 1e-10)
    mIoU_2d = np.mean(iou_class_2d)
    mAcc_2d = np.mean(accuracy_class_2d)
    allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)
    # acc_cls = acc / total

    iou_class_3dmat = intersection_meter_3dmat.sum / (union_meter_3dmat.sum + 1e-10)
    accuracy_class_3dmat = intersection_meter_3dmat.sum / (target_meter_3dmat.sum + 1e-10)
    mIoU_3dmat = np.mean(iou_class_3dmat)
    mAcc_3dmat = np.mean(accuracy_class_3dmat)
    allAcc_3dmat = sum(intersection_meter_3dmat.sum) / (sum(target_meter_3dmat.sum) + 1e-10)

    # allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)
    acc_cls = acc / total

    iou_class_mat = intersection_meter_mat.sum / (union_meter_mat.sum + 1e-10)
    accuracy_class_mat = intersection_meter_mat.sum / (target_meter_mat.sum + 1e-10)
    mIoU_mat = np.mean(iou_class_mat)
    mAcc_mat = np.mean(accuracy_class_mat)
    allAcc_mat = sum(intersection_meter_mat.sum) / (sum(target_meter_mat.sum) + 1e-10)

    if main_process():
        logger.info(
            'Val result 3d: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3d, mAcc_3d, allAcc_3d))
        logger.info(
            'Val result 2d : mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_2d, mAcc_2d, allAcc_2d))
        logger.info(
            'Val result 2dmat: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_mat, mAcc_mat, allAcc_mat))
        logger.info(
            'Val result 3dmat: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3dmat, mAcc_3dmat, allAcc_3dmat))
        logger.info('Class ACC{:.4f}'.format(acc_cls))
        
    return loss_meter_3d.avg, mIoU_3d, mAcc_3d, allAcc_3d, \
           loss_meter_2d.avg, mIoU_2d, mAcc_2d, allAcc_2d


if __name__ == '__main__':
    main()

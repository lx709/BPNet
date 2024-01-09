import os
import random
import numpy as np
import logging
import argparse

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from os.path import join

from util import config
from metrics import iou
from metrics.compat import BboxEval, BboxEvalGTMat, BboxEvalGTPart

from MinkowskiEngine import SparseTensor, CoordsManager
from util import config
from util.util import AverageMeter, intersectionAndUnionGPU
from tqdm import tqdm

import pickle
import json
import h5py

import pudb

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def worker_init_fn(worker_id):
    random.seed(1463 + worker_id)
    np.random.seed(1463 + worker_id)
    torch.manual_seed(1463 + worker_id)


def get_parser():
    parser = argparse.ArgumentParser(description='BPNet')
    parser.add_argument('--config', type=str, default='config/scannet/bpnet_5cm.yaml', help='config file')
    parser.add_argument('opts', help='see config/scannet/bpnet_5cm.yaml for all options', default=None,
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

def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    # https://github.com/Microsoft/human-pose-estimation.pytorch/issues/8
    # https://discuss.pytorch.org/t/training-performance-degrades-with-distributeddataparallel/47152/7
    # torch.backends.cudnn.enabled = False

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        # cudnn.benchmark = False
        # cudnn.deterministic = True

    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False


    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    model = get_model(args)
    if main_process():
        global logger
        logger = get_logger()
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
        args.test_workers = int(args.test_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = model.cuda()

    if os.path.isfile(args.model_path):
        if main_process():
            logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
        if args.multiprocessing_distributed:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(new_state_dict, strict=True)
        if main_process():
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model_path))
#         raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))

    # ####################### Data Loader ####################### #
    if args.data_name == '3dcompat':
        
        import webdataset as wds
        from dataset.GCR_loader import GCRLoader3D

        ROOT_URL = args.data_root
        
        args.com = 10 # always evaluate on 10 compositions
        
        # val_loader = GCRLoader3D(root_url=ROOT_URL, split='valid', n_comp=args.com, view_type=args.view_type, sem_level=args.sem_level).make_loader(batch_size=args.test_batch_size, num_workers=args.workers, aug=args.aug, voxelSize=args.voxelSize, distributed=args.distributed, world_size=args.world_size)
        
        test_loader = GCRLoader3D(root_url=ROOT_URL, split='test', n_comp=args.com, view_type=args.view_type, sem_level=args.sem_level).make_loader(batch_size=args.test_batch_size, num_workers=args.test_workers, aug=args.aug, voxelSize=args.voxelSize, distributed=args.distributed, world_size=args.world_size)
        
    else:
        raise Exception('Dataset not supported yet'.format(args.data_name))

    # ####################### Test ####################### #
    print('start evaluation')
    # validate_cross(model, val_loader)
    test_cross_3d(model, test_loader)
    
def validate_cross(model, val_loader):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3dmat, union_meter_3dmat, intersection_meter_3dmat = AverageMeter(), AverageMeter(), AverageMeter()
    target_meter_mat, union_meter_mat, intersection_meter_mat = AverageMeter(), AverageMeter(), AverageMeter()
    
    acc = 0
    total = 0
    model.eval()
    
    pred_parts, gt_parts = [], []
    pred_mats, gt_mats = [], []
    pred_objs, gt_objs = [], []
    model_ids, style_ids, view_ids = [], [], []
    
    with torch.no_grad():
        for i, batch_data in tqdm(enumerate(val_loader)):
            (coords, feat, label_3d, color, label_2d, link, cls, mat, mat3d, model_id, inds_reverse, style_id, view_id) = batch_data
            sinput = SparseTensor(feat.cuda(non_blocking=True), coords)
            color, link = color.cuda(non_blocking=True), link.cuda(non_blocking=True)
            label_3d, label_2d, = label_3d.cuda(non_blocking=True), label_2d.cuda(non_blocking=True)
            cls, mat = cls.cuda(non_blocking=True), mat.cuda(non_blocking=True)
            mat3d = mat3d.cuda(non_blocking=True)

            output_3d, output_2d, output_mat, output_3dmat, output_cls = model(sinput, color, link)
#             print('before output_3d, output_3dmat', output_3d.shape, output_3dmat.shape)
            output_3d = output_3d[inds_reverse, :]
            output_3dmat = output_3dmat[inds_reverse, :]
#             print('after output_3d, output_3dmat', output_3d.shape, output_3dmat.shape, label_3d.shape, mat3d.shape)
            
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
            # accuracy_3d = sum(intersection_meter_3d.val) / (sum(target_meter_3d.val) + 1e-10)

            # ############ 2D ############ #
            # print(output_2d.shape)
            # o2d=output_2d.detach().topk(5, dim=1)[1]
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
            # o3dmat = output_3dmat.detach().topk(5)[1]
            output_3dmat = output_3dmat.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3dmat, mat3d.detach(), args.mat,
                                                                  args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3dmat.update(intersection)
            union_meter_3dmat.update(union)
            target_meter_3dmat.update(target)
            # accuracy_3dmat = sum(intersection_meter_3dmat.val) / (sum(target_meter_3dmat.val) + 1e-10)
            
            # ############ mat ############ #
            # omat=output_mat.detach().topk(5, dim=1)[1]
            output_mat = output_mat.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_mat, mat.detach(), args.mat,
                                                                  args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_mat.update(intersection)
            union_meter_mat.update(union)
            target_meter_mat.update(target)
            # accuracy_mat = sum(intersection_meter_mat.val) / (sum(target_meter_mat.val) + 1e-10)

            # ############ cls ############ #
            ocls = output_cls.detach().topk(5)[1]
            output_cls = output_cls.detach().max(1)[1]
            correct_guessed = output_cls == cls
            cls_b_acc = torch.sum(correct_guessed.double()).item()
            acc += cls_b_acc
            total += output_cls.size(0)
            
            ### save outputs
            bs = color.shape[0]
            pred_parts.append(output_3d.detach_().cpu().view(bs,2048))
            gt_parts.append(label_3d.cpu().view(bs,2048))
            
            pred_mats.append(output_3dmat.detach_().cpu().view(bs,2048))
            gt_mats.append(mat3d.cpu().view(bs,2048))
            
            pred_objs.append(output_cls.detach_().cpu())
            gt_objs.append(cls.detach_().cpu())
            
            model_ids.append(model_id)
            style_ids.append(style_id.detach_().cpu())
            view_ids.append(view_id.detach_().cpu())
            
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
    
    pred_parts = torch.cat(pred_parts).numpy()
    gt_parts = torch.cat(gt_parts).numpy()

    pred_mats = torch.cat(pred_mats).numpy()
    gt_mats = torch.cat(gt_mats).numpy()

    pred_objs = torch.cat(pred_objs).numpy()
    gt_objs = torch.cat(gt_objs).numpy()
    
#     pudb.set_trace()
    model_ids = np.concatenate(model_ids).astype('S6')
    style_ids = torch.cat(style_ids).numpy().astype('uint8')
    view_ids = torch.cat(view_ids).numpy().astype('uint8')
    
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
        
        print('pred_parts.shape, gt_parts.shape, pred_mats.shape, gt_mats.shape', pred_parts.shape, gt_parts.shape, pred_mats.shape, gt_mats.shape)
        
        data = {'pred_parts': pred_parts, 'gt_parts': gt_parts, 'pred_mats': pred_mats, 'gt_mats': gt_mats, 'pred_objs': pred_objs, 'gt_objs': gt_objs, 'model_ids':model_ids, 'style_ids': style_ids, 'view_ids': view_ids}

        with h5py.File(os.path.join(args.save_folder, 'val_outputs.h5'), 'w') as f:
            for name, arr in data.items():
                f.create_dataset(name, data=arr)
        print('save results done')

def test_cross_3d(model, val_data_loader):
    
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    intersection_meter_3d, intersection_meter_2d = AverageMeter(), AverageMeter()
    union_meter_3d, union_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3d, target_meter_2d = AverageMeter(), AverageMeter()
    target_meter_3dmat, union_meter_3dmat, intersection_meter_3dmat = AverageMeter(), AverageMeter(), AverageMeter()
    target_meter_mat, union_meter_mat, intersection_meter_mat = AverageMeter(), AverageMeter(), AverageMeter()
    
    acc = 0
    total = 0
    model.eval()
    
    pred_parts, gt_parts = [], []
    pred_mats, gt_mats = [], []
    pred_objs, gt_objs = [], []
    model_ids, style_ids, view_ids = [], [], []
    
    with torch.no_grad():
        
        for i, batch_data in enumerate(val_data_loader):

            (coords, feat, label_3d, color, label_2d, link, cls, mat, mat3d, model_id, inds_reverse, style_id, view_id) = batch_data
            # print('feat, coords', feat, coords)
            sinput = SparseTensor(feat.cuda(), coords.cuda())
            color, link = color.cuda(), link.cuda()
            label_3d, label_2d, = label_3d.cuda(), label_2d.cuda()
            cls, mat = cls.cuda(), mat.cuda()
            mat3d = mat3d.cuda()

            output_3d, output_2d, output_mat, output_3dmat, output_cls = model(sinput, color, link)

            output_3d = output_3d[inds_reverse, :]
            output_3dmat = output_3dmat[inds_reverse, :]
            
            # ############ part ############ #
            output_3d = output_3d.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3d, label_3d.detach(), args.classes,
                                                                  args.ignore_label)
            # import pudb; pudb.set_trace()
            # print('before merge', i, coords.shape, color.shape, output_3d.shape, label_3d.shape)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3d.update(intersection)
            union_meter_3d.update(union)
            target_meter_3d.update(target)
            
            
            # ############ mat_3d ############ #
            # o3dmat = output_3dmat.detach().topk(5)[1]
            output_3dmat = output_3dmat.detach().max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output_3dmat, mat3d.detach(), args.mat,
                                                                  args.ignore_label)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter_3dmat.update(intersection)
            union_meter_3dmat.update(union)
            target_meter_3dmat.update(target)
            # accuracy_3dmat = sum(intersection_meter_3dmat.val) / (sum(target_meter_3dmat.val) + 1e-10)
            

            # ############ cls ############ #
            ocls = output_cls.detach().topk(5)[1]
            output_cls = output_cls.detach().max(1)[1]
            correct_guessed = output_cls == cls
            cls_b_acc = torch.sum(correct_guessed.double()).item()
            acc += cls_b_acc
            total += output_cls.size(0)
            
            ### save outputs
            bs = color.shape[0]
            pred_parts.append(output_3d.detach_().cpu().view(bs,2048))
            gt_parts.append(label_3d.cpu().view(bs,2048))
            
            pred_mats.append(output_3dmat.detach_().cpu().view(bs,2048))
            gt_mats.append(mat3d.cpu().view(bs,2048))
            
            pred_objs.append(output_cls.detach_().cpu())
            gt_objs.append(cls.detach_().cpu())
            
            model_ids.append(model_id)
            style_ids.append(style_id.detach_().cpu())
            view_ids.append(view_id.detach_().cpu())
        
        iou_class_3d = intersection_meter_3d.sum / (union_meter_3d.sum + 1e-10)
        accuracy_class_3d = intersection_meter_3d.sum / (target_meter_3d.sum + 1e-10)
        mIoU_3d = np.mean(iou_class_3d)
        mAcc_3d = np.mean(accuracy_class_3d)
        allAcc_3d = sum(intersection_meter_3d.sum) / (sum(target_meter_3d.sum) + 1e-10)


        iou_class_3dmat = intersection_meter_3dmat.sum / (union_meter_3dmat.sum + 1e-10)
        accuracy_class_3dmat = intersection_meter_3dmat.sum / (target_meter_3dmat.sum + 1e-10)
        mIoU_3dmat = np.mean(iou_class_3dmat)
        mAcc_3dmat = np.mean(accuracy_class_3dmat)
        allAcc_3dmat = sum(intersection_meter_3dmat.sum) / (sum(target_meter_3dmat.sum) + 1e-10)

        # allAcc_2d = sum(intersection_meter_2d.sum) / (sum(target_meter_2d.sum) + 1e-10)
        acc_cls = acc / total
        
        pred_parts = torch.cat(pred_parts).numpy()
        gt_parts = torch.cat(gt_parts).numpy()
        
        pred_mats = torch.cat(pred_mats).numpy()
        gt_mats = torch.cat(gt_mats).numpy()
        
        pred_objs = torch.cat(pred_objs).numpy()
        gt_objs = torch.cat(gt_objs).numpy()
        
        model_ids = np.concatenate(model_ids).astype('S6')
        style_ids = torch.cat(style_ids).numpy().astype('uint8')
        view_ids = torch.cat(view_ids).numpy().astype('uint8')

        if main_process():
            
            logger.info(
                'Val result 3d: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3d, mAcc_3d, allAcc_3d))
            logger.info(
                'Val result 3dmat: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU_3dmat, mAcc_3dmat, allAcc_3dmat))
            logger.info('Class ACC{:.4f}'.format(acc_cls))
            
            print('pred_parts.shape, gt_parts.shape, pred_mats.shape, gt_mats.shape', pred_parts.shape, gt_parts.shape, pred_mats.shape, gt_mats.shape)
            
            data = {'pred_parts': pred_parts, 'gt_parts': gt_parts, 'pred_mats': pred_mats, 'gt_mats': gt_mats, 'pred_objs': pred_objs, 'gt_objs': gt_objs, 'model_ids':model_ids, 'style_ids': style_ids, 'view_ids': view_ids}

            os.makedirs(args.save_folder, exist_ok=True)
            save_path = os.path.join(args.save_folder, 'test_outputs_2.h5')
            if os.path.exists(save_path):
                # remove file
                os.system('rm {}'.format(save_path))
                
            with h5py.File(save_path, 'w') as f:
                for name, arr in data.items():
                    f.create_dataset(name, data=arr)
            print(f'save results to {save_path}')
        
if __name__ == '__main__':
    main()

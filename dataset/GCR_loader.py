""""
Dataloaders for the 2D 3DCoMPaT tasks.
"""
import json
import os
import webdataset as wds
import re
import numpy as np
import math
from functools import partial

from . import augmentation as t_3d
from . import augmentation_2d as t_2d
from .voxelizer import Voxelizer

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pudb

import json
import h5py
import os

from .utils import masks as masks
from .utils import regex as regex
from .utils import stream as stream
from .utils import depth as depth

def wds_identity(x):
    return x

SHUFFLE_CONFIG = {
    "train": 0,
    "valid": 0,
    "test": 0
}


class CompatLoader2D:
    """
    Base class for 2D dataset loaders.

    Args:
        root_url:    Base dataset URL containing data split shards
        split:       One of {train, valid}
        sem_level:   Semantic level. One of {fine, coarse}
        n_comp:      Number of compositions to use
        cache_dir:   Cache directory to use
        view_type:   Filter by view type [0: canonical views, 1: random views]
        transform:   Transform to be applied on rendered views
    """
    def __init__(self, root_url, split, sem_level, n_comp, cache_dir=None, view_type=-1, transform=wds_identity):
        valid_splits = ["train", "valid", "test"]
        valid_sem_levels = ["coarse", "fine"]

        if view_type not in [-1, 0, 1]:
            raise RuntimeError("Invalid argument: view_type can only be [-1, 0, 1]")
        if split not in valid_splits:
            raise RuntimeError("Invalid split: [%s]." % split)
        if sem_level not in valid_sem_levels:
            raise RuntimeError("Invalid semantic level: [%s]." % split)

        if root_url[-1] == '/':
            root_url = root_url[:-1]

        # Reading sample count from metadata
        datacount_file = root_url + "/datacount.json"
        if regex.is_url(root_url):
            # Optionally: downloading the datacount file over the Web
            os.system("wget -O %s %s >/dev/null 2>&1" % ("./datacount.json", datacount_file))
            datacount = json.load(open("./datacount.json", "r"))
        else:
            datacount = json.load(open(datacount_file, "r"))
        sample_count = datacount['sample_count']
        max_comp     = datacount['compositions']

        if n_comp > max_comp:
            except_str = "Required number of compositions exceeds maximum available in [%s] (%d)." % (root_url, n_comp)
            raise RuntimeError(except_str)

        # Computing dataset size
        self.dataset_size = sample_count[split]*n_comp
        if view_type != -1:
            self.dataset_size //= 2

        # Configuring size of shuffle buffer
        self.shuffle = SHUFFLE_CONFIG[split]

        # Formatting WebDataset base URL
        comp_str = "%04d" % (n_comp -1)
        self.url = '%s/%s/%s_%s_{0000..%s}.tar' % (root_url,
                                                   split,
                                                   split,
                                                   sem_level,
                                                   comp_str)

        self.sem_level = valid_sem_levels.index(sem_level)
        self.cache_dir = cache_dir
        self.transform = transform
        self.view_type = view_type


    def make_loader(self):
        """
        Instantiating dataloader

        Args:
            batch_size:  Size of each batch in the loader
            num_workers: Number of process workers to use when loading data
        """
        # Instantiating dataset
        dataset = wds.WebDataset(self.url, cache_dir=self.cache_dir)

        if self.view_type != -1:
            view_val = bytes(str(self.view_type), 'utf-8')
            dataset = dataset.select(lambda x: x["view_type.cls"] == view_val)
        if self.shuffle > 0:
            dataset = dataset.shuffle(self.shuffle)

        return dataset

class FullLoader(CompatLoader2D):
    """
    Dataloader for the full data available in the WDS shards.
    Adapt and filter to the fields needed for your usage.

    Args:
        -> CompatLoader2D

        mask_transform:  Transform to apply to segmentation masks
    """

    def __init__(self, root_url, split, sem_level, n_comp, cache_dir=None, view_type=-1,
                    transform=wds_identity,
                    mask_transform=wds_identity,
                    part_mat_transform=wds_identity,
                    depth_transform=wds_identity):
        super().__init__(root_url, split, sem_level, n_comp, cache_dir, view_type, transform)

        self.mask_transform  = partial(masks.mask_decode_partial, mask_transform, [0, 1])
        self.depth_transform = partial(depth.depth_decode, depth_transform)
        self.split_masks = partial(stream.split_masks, 3)
        self.part_mat_transform = part_mat_transform

    def make_loader(self, batch_size, num_workers):
        # Instantiating dataset
        dataset = (
            super().make_loader()
            .decode(
                wds.handle_extension("mask.png", self.mask_transform),
                wds.imagehandler("torchrgb")
            )
            .to_tuple("model_id.txt",
                      "render.png",
                      "target.cls",
                      "mask.png",
                      "depth.exr",
                      "style_id.cls",
                      "view_id.cls",
                      "view_type.cls",
                      "cam.ten")
            .map_tuple(wds_identity,
                       self.transform,
                       wds_identity,
                       wds_identity,
                       self.depth_transform,
                       wds_identity,
                       wds_identity,
                       wds_identity,
                       stream.unwrap_cam)
            .batched(batch_size, partial=False)
        ).compose(self.split_masks)

        # Instantiating loader
        loader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
        )

        # Defining loader length
        loader.length = self.dataset_size // batch_size

        return loader
    
#### for GCR 3D task

# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


class LinkCreator(object):
    def __init__(self, fx=-284.44, fy=284.44, mx=128, my=128, image_dim=(256, 256), voxelSize=0.05):
        self.intricsic = make_intrinsic(fx=fx, fy=fy, mx=mx, my=my)
        self.imageDim = image_dim
        self.voxel_size = voxelSize

    def computeLinking(self, world_to_camera, coords, depth):
        """
        three different points from original bpnet: 1) input camera matrix is world_to_camera not camera_to_world; 2) no need to use minus values for p[1]; 3) p[2] are all minus values.
        
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :return: linking, N x 3 format, (H,W,mask)
        """
        link = np.zeros((3, coords.shape[0]), dtype=int)
        coordsNew = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coordsNew.shape[0] == 4, "[!] Shape error"
#         world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coordsNew)
        p[0] = (p[0] * self.intricsic[0][0]) / p[2] + self.intricsic[0][2]
        p[1] = ((p[1] * self.intricsic[1][1]) / p[2] + self.intricsic[1][2])
        pi = np.round(p).astype(int)
        inside_mask = (pi[0] >= 0) * (pi[1] >= 0) \
                      * (pi[0] < self.imageDim[0]) * (pi[1] < self.imageDim[1])
        occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                + p[2][inside_mask]) <= self.voxel_size
#         print('inside_mask, occlusion_mask', np.sum(inside_mask), np.sum(occlusion_mask))
        inside_mask[inside_mask == True] = occlusion_mask
        link[0][inside_mask] = pi[1][inside_mask]
        link[1][inside_mask] = pi[0][inside_mask]
        link[2][inside_mask] = 1
        return link.T
    

    
def get_transform_2d(aug=True, IMG_DIM=(256,256), value_scale=1.0):
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    if aug:
        transform_2d = t_2d.Compose([
            t_2d.RandomGaussianBlur(),
            t_2d.Crop([IMG_DIM[1], IMG_DIM[0]], crop_type='rand', padding=mean,
                      ignore_label=255),
            t_2d.ToTensor(),
            t_2d.Normalize(mean=mean, std=std)])
    else:
        transform_2d = t_2d.Compose([
            t_2d.Crop([IMG_DIM[1], IMG_DIM[0]], crop_type='rand', padding=mean,
                      ignore_label=255),
            t_2d.ToTensor(),
            t_2d.Normalize(mean=mean, std=std)])
    
    return transform_2d

    
def get_transforms_3d(voxelSize=0.05):
    
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi,
                                                                                          np.pi))
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = 'z'
    LOCFEAT_IDX = 2
    
    data_aug_color_trans_ratio=0.1
    data_aug_color_jitter_std=0.05
    data_aug_hue_max=0.5
    data_aug_saturation_max=0.2
    
    voxelizer = Voxelizer(
            voxel_size=voxelSize,
            clip_bound=None,
            use_augmentation=True,
            scale_augmentation_bound=SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=TRANSLATION_AUGMENTATION_RATIO_BOUND)
    
    
    prevoxel_transform_train = [t_3d.ElasticDistortion(ELASTIC_DISTORT_PARAMS)]
    prevoxel_transforms = t_3d.Compose(prevoxel_transform_train)
    input_transforms = [
        t_3d.RandomHorizontalFlip(ROTATION_AXIS, is_temporal=False),
        t_3d.ChromaticAutoContrast(),
        t_3d.ChromaticTranslation(data_aug_color_trans_ratio),
        t_3d.ChromaticJitter(data_aug_color_jitter_std),
        t_3d.HueSaturationTranslation(data_aug_hue_max, data_aug_saturation_max),
    ]
    input_transforms = t_3d.Compose(input_transforms)
    
    return voxelizer, prevoxel_transforms, input_transforms
    

def collation_fn(batch, V=8, eval_all=False):
    """
    :param batch:
    input: 
        colors: B x V x 3 x H x W
        part_2d: B x V x H x W
        links: B x N_MAX x 4 x V
        coords: B x N_MAX x 4

    :return:    coords: N_all x 4 (batch,x,y,z)
                feats:  N_all x 3
                labels: N
                colors: B x C x H x W x V
                labels_2d:  B x H x W x V
                links:  N_all x 4 x V (B,H,W,mask)

    """
    colors, part_2d, links, obj_cls, mat_2d, coords, feats, part_mat_3d, model_id, NP, inds_recons, style_id, view_id = batch
    
    ## print('collation input:', colors.shape, part_2d.shape, mat_2d.shape, links.shape, feats.shape, part_mat_3d.shape)
    
    ## put V dim to the end
    colors = colors.view(-1,V,colors.shape[-3], colors.shape[-2], colors.shape[-1]).permute(0,2,3,4,1)
    part_2d = part_2d.view(-1,V,part_2d.shape[-2], part_2d.shape[-1]).permute(0,2,3,1)
    mat_2d = mat_2d.view(-1,V,mat_2d.shape[-2], mat_2d.shape[-1]).permute(0,2,3,1)
    obj_cls = torch.from_numpy(obj_cls).long().view(-1,V)[:,0] # only need one label for all views

    inds_recons = inds_recons.float()
    coords_all = []
    feat_all = []
    part_3d_all = []
    mat_3d_all = []
    links_all = []
    inds_recons_all = []
    
    accmulate_points_num = 0
    for i in range(len(coords)):
        coords[i][:, 0] = i # need to set to batch_idx
        links[i][:, 0, :] = i

        coords_all.append(coords[i][:NP[i]])
        feat_all.append(feats[i][:NP[i]])
        links_all.append(links[i][:NP[i]])
        
        if not eval_all:
            part_3d_all.append(part_mat_3d[i][:NP[i],0])
            mat_3d_all.append(part_mat_3d[i][:NP[i],1])
        else:
            part_3d_all.append(part_mat_3d[i][:,0])
            mat_3d_all.append(part_mat_3d[i][:,1])
            
            inds_recons_all.append(accmulate_points_num + inds_recons[i])
            accmulate_points_num += NP[i] # do not use len(inds_recons[i])
    
    if eval_all:
        inds_recons_all = torch.cat(inds_recons_all).long()
    
#     print('collation output: colors, coords_all, link', colors.shape, torch.cat(coords_all).shape, torch.cat(links_all).shape)
    
    return torch.cat(coords_all).int(), torch.cat(feat_all).float(), torch.cat(part_3d_all).long(), \
           colors.float(), part_2d.long(), torch.cat(links_all), obj_cls, mat_2d.long(), torch.cat(mat_3d_all).long(), model_id, inds_recons_all, style_id, view_id

    
class GCRLoader3D(CompatLoader2D):
    """
    Dataloader for the full 2D compositional task.
    Iterating over 2D renderings of shapes with:
        - shape category label
        - segmentation mask with pixel coding for parts
        - part materials labels

    Args:
        -> CompatLoader2D

        mask_transform:      Transform to apply on segmentation masks
        code_transform:      Function to apply on segmentation mask labels 
        part_mat_transform:  Function to apply on part-material labels
    """

    def __init__(self, root_url, split, sem_level, n_comp, cache_dir=None, view_type=-1,
                    transform=wds_identity,
                    mask_transform=wds_identity,
                    part_mat_transform=wds_identity,
                    depth_transform=wds_identity):
        super().__init__(root_url, split, sem_level, n_comp, cache_dir, view_type, transform)
        
        self.mask_transform  = partial(masks.mask_decode_partial, mask_transform, [0, 1])
        self.depth_transform = partial(depth.depth_decode, depth_transform)
        self.split_masks = partial(stream.split_masks, 3)
        self.part_mat_transform = part_mat_transform
        self.split = split
        
        self.load_point_clouds(sem_level)
        
        if view_type==-1:
            self.view_num = 8
        else:
            self.view_num = 4
        self.length = None

    def load_point_clouds(self, sem_level = 'coarse'):
        
        hdf_path = '/lustre/scratch/project/k1546/3DCoMPaT-v2/hdf5/'

        data = h5py.File(hdf_path + '{}_{}.hdf5'.format(self.split, sem_level))

        self.all_shape_ids = np.array(data['shape_id']).astype('str')
        self.all_style_ids = np.array(data['style_id']).astype('uint8')
        self.all_points = np.array(data['points']).astype('float16')
        self.all_obj_label = np.array(data['shape_label']).astype('uint8')
        self.all_part_labels = np.array(data['points_part_labels']).astype('uint16')
        self.all_mat_labels = np.array(data['points_mat_labels']).astype('uint8')
    
    def load_point_cloud(self, model_id, style_id):

        idx = np.where((self.all_shape_ids == model_id) & (self.all_style_ids == style_id))[0][0]

        if idx is None:
            assert 'cannot find a matched 3D mdoel'
        locs_in = self.all_points[idx][:,:3]
        feats_in = self.all_points[idx][:,3:]
#         feats_in = np.random.randn(locs_in.shape[0],3)
        part_3d = self.all_part_labels[idx] + 1 # add background classes
        mat_3d = self.all_mat_labels[idx] + 1 # add background classes
        labels_in = np.stack([part_3d, mat_3d], -1)

        return locs_in, feats_in, labels_in

    # Define a function to process each item in the dataset
    def process_item(self, item, voxelSize=0.05, aug=True):
        '''
        Args:
            voxelSize=0.05,
            aug=True

        returns:
            colors: C x H x W
            labels_2d:  1 X H x W
            mat_2d:  1 X H x W
            links:  N x 4
            model_id
        '''

        #### preparing 2D data

        model_id, colors, obj_cls, part_mat_2d, depth, style_id, view_id, view_type, pose = item
        IMG_DIM = (colors.shape[-2],colors.shape[-1])

        # Load the corresponding point cloud
        locs_in, feats_in, labels_in = self.load_point_cloud(model_id, style_id)
        coords = locs_in.copy()
        coords[:,2] = coords[:,2] - coords[:,2].min() # move up z-axis

        ## tensor to numpy
        colors = colors.permute(1,2,0).numpy()
        part_mat_2d = part_mat_2d.permute(1,2,0).numpy()

        links = np.ones([coords.shape[0], 4], dtype=np.int64)
        linkCreator = LinkCreator(fx=-284.44, fy=284.44, mx=128, my=128, voxelSize=voxelSize, image_dim=IMG_DIM)
        links[:, 1:4] = linkCreator.computeLinking(pose[4:,:], coords, depth)
        # print('valid links', np.sum(links[:,-1]))
        links = torch.from_numpy(links)

        transform_2d = get_transform_2d(aug=aug, IMG_DIM=IMG_DIM, value_scale=1.0) # scaled to 1 when decode in webdataset
        colors, part_mat_2d = transform_2d(colors, part_mat_2d)
        part_2d, mat_2d = torch.split(part_mat_2d, 1, dim=-1)
        part_2d, mat_2d = part_2d.squeeze(), mat_2d.squeeze()

    #     print(type(coords), coords.shape, part_2d.shape, mat_2d.shape, links.shape)
        return locs_in, colors, part_2d, links, obj_cls, mat_2d, model_id, locs_in, feats_in, labels_in, style_id, view_id

    def collation_view(self, batch, V=8, voxelSize=0.05, aug=True, eval_all=False):
        """
        :param batch:
        input: 
            colors: V x [C x H x W]
            part_2d: V x [H x W]
            links: V x [N x 4]

        :return:    
                    colors: V x C x H x W
                    part_2d:  V x H x W
                    mat_2d:  V x H x W
                    links:  N x 4 x V (H,W,mask)
                    coords: N x 4 (batch,x,y,z)
                    feats:  N x 3
                    part_mat_3d: N x 2

        """

        locs_in, colors, part_2d, links, obj_cls, mat_2d, model_id, locs_in, feats_in, labels_in, style_id, view_id = batch

        locs_in, feats_in, labels_in = locs_in[0], feats_in[0], labels_in[0]
        MAX_NUM_POINTS = locs_in.shape[0] * 1 # maximum points filtered from different view points

        links = links.permute(1,2,0)

        #### preparing 3D data
        voxelizer, prevoxel_transforms, input_transforms = get_transforms_3d(voxelSize=voxelSize)
        locs = prevoxel_transforms(locs_in) if aug else locs_in
        locs, feats, part_mat_3d, inds_recons, links = voxelizer.voxelize(locs, feats_in, labels_in, link=links)
    #     print('feats_in, feats, inds_recons, links', feats_in.shape, feats.shape, inds_recons.shape, links.shape)

        if eval_all:
            part_mat_3d = labels_in
    #         print(eval_all, part_mat_3d.shape)

        if aug:
            locs, feats, part_mat_3d = input_transforms(locs, feats, part_mat_3d)

        coords = torch.from_numpy(locs).int()
        coords = torch.cat((torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
        feats = torch.from_numpy(feats).float() / 127.5 - 1 #normalized to [-1,1]
        part_mat_3d = torch.from_numpy(part_mat_3d.astype('int')).long()
        inds_recons = torch.from_numpy(inds_recons).int()

        ### padding point clouds to fixed number of points, to enable batch data processing
        N = coords.shape[0]
        assert N<=MAX_NUM_POINTS
        coords = F.pad(coords, (0, 0, 0, MAX_NUM_POINTS-N), mode='constant', value=-1)
        feats = F.pad(feats, (0, 0, 0, MAX_NUM_POINTS-N), mode='constant', value=-1)
        if not eval_all:
            part_mat_3d = F.pad(part_mat_3d, (0, 0, 0, MAX_NUM_POINTS-N), mode='constant', value=-1)
        links = F.pad(links, (0, 0, 0, 0, 0, MAX_NUM_POINTS-N), mode='constant', value=-1)

    #     print(colors.shape, part_2d.shape, links.shape, obj_cls, mat_2d.shape, coords.shape, feats.shape, part_mat_3d.shape, model_id, N)

        return colors, part_2d, links, obj_cls, mat_2d, coords, feats, part_mat_3d, model_id, N, inds_recons, style_id, view_id

    def __len__(self):
        if self.length is None:
            # Count the number of samples in the dataset
            self.length = sum(1 for _ in self.dataset)
        return self.length

    def __iter__(self):
        return iter(self.dataset)
    
    def make_loader(self, batch_size=1, num_workers=1, aug=False, voxelSize=0.05, distributed=False, world_size=1):
        # Instantiating dataset

        if self.split=='train':
            eval_all = False
        else:
            eval_all = True
            
        dataset = (
            super().make_loader()
            .decode(
                wds.handle_extension("mask.png", self.mask_transform),
                wds.imagehandler("torchrgb")
            )
            .to_tuple("model_id.txt",
                      "render.png",
                      "target.cls",
                      "mask.png",
                      "depth.exr",
                      "style_id.cls",
                      "view_id.cls",
                      "view_type.cls",
                      "cam.ten")
            .map_tuple(wds_identity,
                       self.transform,
                       wds_identity,
                       wds_identity,
                       self.depth_transform,
                       wds_identity,
                       wds_identity,
                       wds_identity,
                       stream.unwrap_cam)
            .map(partial(self.process_item, aug=aug, voxelSize=voxelSize))
            .batched(self.view_num, partial=False)
            .map(partial(self.collation_view, V=self.view_num, voxelSize=voxelSize,aug=aug,eval_all=eval_all))
            .batched(batch_size, partial=False)
            .map(partial(collation_fn, V=self.view_num,eval_all=eval_all))
            .shuffle(True)
        )
        
        # WebLoader is just the regular DataLoader with the same convenience methods
        # that WebDataset has.
        loader = wds.WebLoader(
            dataset, batch_size=None, shuffle=False, num_workers=num_workers,
        )
        if distributed:
            # With DDP, we need to make sure that all nodes get the same number of batches;
            # we do that by reusing a little bit of data.
            # Note that you only need to do this when retrofitting code that depends on
            # epoch size. A better way is to iterate through the entire dataset on all nodes.
            dataset_size = self.dataset_size
            number_of_batches = dataset_size // (batch_size * world_size * self.view_num)  ### Note that batch_size is the number of shapes
            print('{} dataset_size: '.format(self.split), dataset_size, batch_size, world_size, self.view_num)
            print("# batches per node = ", number_of_batches)
            loader = loader.repeat(2).slice(number_of_batches) # If dataset_size can be divided by (batch_size * world_size), then this do nothing, it works like keep_last in torch dataloader
            # This only sets the value returned by the len() function; nothing else uses it,
            # but some frameworks care about it.
            loader.length = number_of_batches
            print('loader.length', loader.length)
        return loader


# This script is modfied from :
# https://github.com/binli123/dsmil-wsi/blob/master/compute_feats.py
# https://github.com/binli123/dsmil-wsi/blob/master/deepzoom_tiler.py 


import os
import ctypes

import json
import concurrent.futures
from multiprocessing import Process, JoinableQueue
import argparse

import re
import shutil
import sys
import glob
import copy
import numpy as np
import math
from unicodedata import normalize
from skimage import io, color, util, filters
from PIL import Image, ImageFilter, ImageStat
import sh
aws = sh.aws
import timm

Image.MAX_IMAGE_PIXELS = None

import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
import pandas as pd

import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

from collections import OrderedDict
from sklearn.utils import shuffle
from concurrent.futures import ThreadPoolExecutor

import utils
import vision_transformer as vits

VIEWER_SLIDE_NAME = 'slide'

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                quality, threshold):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            try:
                tile = dz.get_tile(level, address)
                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge = ImageStat.Stat(edge).sum
                edge = np.mean(edge)/(self._tile_size**2)
                w, h = tile.size
                if edge > self._threshold:
                    if not (w==self._tile_size and h==self._tile_size):
                        tile = tile.resize((self._tile_size, self._tile_size))
                    tile.save(outfile, quality=self._quality)
            except:
                pass
            self._queue.task_done()
            

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a tcga image."""

    def __init__(self, dz, basename, target_levels, mag_base, format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._target_levels = target_levels
        self._mag_base = int(mag_base)

    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        target_levels = [self._dz.level_count-i-1 for i in self._target_levels]
        mag_list = [int(self._mag_base/2**i) for i in self._target_levels]
        mag_idx = 0
        for level in range(self._dz.level_count):
            if not (level in target_levels):
                continue
            tiledir = os.path.join("%s_files" % self._basename, str(mag_list[mag_idx]))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            cols, rows = self._dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                    col, row, self._format))
                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row),
                                    tilename))
                    self._tile_done()
            mag_idx += 1

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, mag_levels, base_mag, objective, format, tile_size, overlap,
                limit_bounds, quality, workers, threshold):
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mag = base_mag
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._dzi_data = {}
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality, threshold).start()

    def run(self):
        self._run_image()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a tcga image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        
        MAG_BASE = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if MAG_BASE is None:
            MAG_BASE = self._objective
        first_level = int(math.log2(float(MAG_BASE)/self._base_mag)) # raw / input, 40/20=2, 40/40=0
        target_levels = [i+first_level for i in self._mag_levels] # levels start from 0
        target_levels.reverse()
        
        tiler = DeepZoomImageTiler(dz, basename, target_levels, MAG_BASE, self._format, associated,
                    self._queue)
        tiler.run()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

def nested_patches(img_slide, out_base,WSI_temp, level=(0,), ext='jpeg'):
    print('\n Organizing patches')
    img_name = img_slide.split(os.sep)[-1]
    img_class = img_slide.split(os.sep)[2]
    n_levels = len(glob.glob(WSI_temp + '/*'))
    bag_path = os.path.join(out_base, img_class, img_name)
    os.makedirs(bag_path, exist_ok=True)
    if len(level)==1:
        patches = glob.glob(os.path.join(WSI_temp , '*', '*.'+ext))
        for i, patch in enumerate(patches):
            patch_name = patch.split(os.sep)[-1]
            shutil.move(patch, os.path.join(bag_path, patch_name))
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(patches)))
        print('Done.')
    else:
        level_factor = 2**int(level[1]-level[0])
        levels = [int(os.path.basename(i)) for i in glob.glob(os.path.join(WSI_temp , '*'))]
        levels.sort()
        low_patches = glob.glob(os.path.join(WSI_temp, str(levels[0]), '*.'+ext))
        for i, low_patch in enumerate(low_patches):
            low_patch_name = low_patch.split(os.sep)[-1]
            shutil.move(low_patch, os.path.join(bag_path, low_patch_name))
            low_patch_folder = low_patch_name.split('.')[0]
            high_patch_path = os.path.join(bag_path, low_patch_folder)
            os.makedirs(high_patch_path, exist_ok=True)
            low_x = int(low_patch_folder.split('_')[0])
            low_y = int(low_patch_folder.split('_')[1])
            high_x_list = list( range(low_x*level_factor, (low_x+1)*level_factor) )
            high_y_list = list( range(low_y*level_factor, (low_y+1)*level_factor) )
            for x_pos in high_x_list:
                for y_pos in high_y_list:
                    high_patch = glob.glob(os.path.join(WSI_temp , str(levels[1]), '{}_{}.'.format(x_pos, y_pos)+ext))
                    if len(high_patch)!=0:
                        high_patch = high_patch[0]
                        shutil.move(high_patch, os.path.join(bag_path, low_patch_folder, high_patch.split(os.sep)[-1]))
            try:
                os.rmdir(os.path.join(bag_path, low_patch_folder))
                os.remove(low_patch)
            except:
                pass
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(low_patches)))
        print('Done.')


class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
      
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        try:
            img = Image.open(img)
        except:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img} 
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

from PIL import Image


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            path, _ = self.samples[index]
            sample = self.loader(path)
            
            # Apply the transform
            if self.transform is not None:
                sample = self.transform(sample)
            
            return sample, self.targets[index]
        except:
            print(f"Error loading image {path}")
            return torch.zeros((3, 224, 224)), -1

def compute_feats(args, bags_list, model, save_path=None, magnification='single'):
    model.eval()
    transform = transforms.ToTensor()
    dataset = CustomImageFolder(bags_list, transform=transform)

    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")
    feats_list = []
    with torch.no_grad():
        for iteration, batch in enumerate(dataloader):
            
            images = [im.cuda(non_blocking=True) for im in batch]
            feats = model(images[0]).clone()
            feats = feats.cpu().numpy()   
            feats_list.extend(feats)
            sys.stdout.write('\r Computed: -- {}/{}'.format(iteration+1, len(dataloader)))
    
    df = pd.DataFrame(feats_list)
    os.makedirs(os.path.join(save_path, args.feats_path, args.backbone), exist_ok=True)
    df.to_csv(os.path.join(save_path,args.feats_path, args.backbone, bags_list.split('_files')[0]+'.csv'), index=False, float_format='%.4f')

def remove_file(path):
    try:
        os.unlink(path)
        sys.stdout.write(f"Deleted file: {path}\n")
    except Exception as e:
        print(f"Error deleting file {path}: {e}")

def process_patient(row, levels, args, MAIN_PATH, model, feats_path):
    WSI_temp = row['filename']+ '_files'
    DeepZoomStaticTiler(c_slide, row['filename'], levels, args.base_mag, args.objective, args.format, args.tile_size, args.overlap, True, args.quality, args.workers, args.background_t).run()
    compute_feats(args, WSI_temp, model, feats_path, args.magnifications)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patch extraction for WSI')
    parser.add_argument('--dataset', type=str, default='TCGA-lung', help='Dataset name')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap of adjacent tiles')
    parser.add_argument('--format', type=str, default='jpeg', help='Image format for tiles')
    parser.add_argument('--slide_format', type=str, default='tif', help='Slide image format')
    parser.add_argument('--workers', type=int, default=32, help='Number of worker processes')
    parser.add_argument('--quality', type=int, default=70, help='JPEG compression quality')
    parser.add_argument('--tile_size', type=int, default=224, help='Tile size')
    parser.add_argument('--base_mag', type=float, default=20, help='Base magnification for patch extraction')
    parser.add_argument('--magnifications', type=int, nargs='+', default=(0,), help='Levels for patch extraction')
    parser.add_argument('--objective', type=float, default=20, help='Default objective power if not in metadata')
    parser.add_argument('--background_t', type=int, default=15, help='Threshold for filtering background')  
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for dataloader')
    parser.add_argument('--num_workers', default=32, type=int, help='Number of threads for dataloader')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model')
    parser.add_argument('--main_path', type=str, default='', help='Main path for dataset')
    parser.add_argument('--feats_path', type=str, default='feats_tcga_rcc', help='Path for storing features')
    parser.add_argument('--stages_file', type=str, default='kirp.txt', help='Stages file name')
    parser.add_argument('--backbone', default='ssl_vit', type=str, choices=['ssl_vit', 'resnet50'], help='Backbone model')

    args = parser.parse_args()
    levels = tuple(sorted(args.magnifications))
    assert len(levels) <= 2, 'Only 1 or 2 magnifications are supported!'

    MAIN_PATH = args.main_path
    feats_path = os.path.join(MAIN_PATH, args.feats_path)
    stages = pd.read_csv(os.path.join(MAIN_PATH, args.stages_file), delimiter='\s+', index_col=False)

    if args.backbone == 'ssl_vit':
        model = timm.create_model('vit_small_patch16_224', pretrained=True).cuda()
    elif args.backbone == 'resnet50':
        norm_layer = nn.BatchNorm2d if pretrain else None
        model = models.resnet50(pretrained='ImageNet', norm_layer=norm_layer).cuda()
    
    model.eval()
    for index, row in stages.iterrows():
        process_patient(row, levels, args, MAIN_PATH, model, feats_path)


       
  


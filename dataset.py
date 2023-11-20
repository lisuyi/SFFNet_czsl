# external libs
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import random
from os.path import join as ospj
from glob import glob
# torch libs
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
# local libs
from utils.utils import get_norm_values, chunks
from models.image_extractor import get_image_extractor
from itertools import product
import re

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ImageLoader:
    def __init__(self, root):
        self.root_dir = root

    def __call__(self, img):
        '''
            If the code run in the linux,the first code below should be commented.
            If the code run in the windows environment,the environment will chang the invalid character into '_',
            the first code below will deal the problem.
            Besides,make sure the computer configuration->file system allow win32-long
            -path, because the file names in the dataset are rather long.
        '''
        # img = re.sub(r'\\|:|\*|\?|"|<|>|\|', '_', img)
        img = Image.open(ospj(self.root_dir, img)).convert('RGB')  # We don't want alpha
        return img  # img = Image.open(ospj(self.root_dir,img)).convert('RGB')


def dataset_transform(phase, norm_family='imagenet'):
    mean, std = get_norm_values(norm_family=norm_family)

    if phase == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    elif phase == 'val' or phase == 'test':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif phase == 'all':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Invalid transform')

    return transform


def filter_data(all_data, pairs_gt, topk=5):
    valid_files = []
    with open('/home/ubuntu/workspace/top' + str(topk) + '.txt') as f:
        for line in f:
            valid_files.append(line.strip())

    data, pairs, attr, obj = [], [], [], []
    for current in all_data:
        if current[0] in valid_files:
            data.append(current)
            pairs.append((current[1], current[2]))
            attr.append(current[1])
            obj.append(current[2])

    counter = 0
    for current in pairs_gt:
        if current in pairs:
            counter += 1
    print('Matches ', counter, ' out of ', len(pairs_gt))
    print('Samples ', len(data), ' out of ', len(all_data))
    return data, sorted(list(set(pairs))), sorted(list(set(attr))), sorted(list(set(obj)))


class CompositionDataset(Dataset):

    def __init__(
            self,
            args,
            root,
            phase,
            split='compositional-split',
            model='resnet18',
            norm_family='imagenet',
            num_negs=1,
            use_precomputed_features=False,
            return_images=False,
            train_only=False,
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.args = args
        self.norm_family = norm_family
        self.return_images = return_images
        self.use_precomputed_features = use_precomputed_features

        if 'resnet18' in model:
            self.feat_dim = 512
        elif 'dino' in model:
            self.feat_dim = 768
        else:
            self.feat_dim = 2048

        self.attrs, self.objs, self.pairs, self.train_pairs, self.val_pairs, self.test_pairs = self.parse_split()
        self.train_data, self.val_data, self.test_data = self.get_split_info()  # 获取文件名
        self.full_pairs = list(product(self.attrs, self.objs))

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}

        self.all_pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        if train_only and self.phase == 'train':
            print('Using only train pairs')
            self.pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}
        else:
            print('Using all dataset pairs：')
            self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        elif self.phase == 'test':
            self.data = self.test_data
        elif self.phase == 'all':
            print('Using all data')
            self.data = self.train_data + self.val_data + self.test_data
        else:
            raise ValueError('Invalid training phase')

        self.all_data = self.train_data + self.val_data + self.test_data
        print('Dataset loaded')
        print('All attrs: {}, All objs: {}, All pairs in dataset: {}'.format(
            len(self.attrs), len(self.objs), len(self.pairs)))
        print('Train pairs: {}, Validation pairs: {}, Test Pairs: {}'.format(
            len(self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('Train images: {}, Validation images: {}, Test images: {}'.format(
            len(self.train_data), len(self.val_data), len(self.test_data)))

        #  Keeping a list of all pairs that occur with each object
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data + self.test_data if obj == _obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj) in self.train_data if obj == _obj]
            self.train_obj_affordance[_obj] = list(set(candidates))

        self.sample_indices = list(range(len(self.data)))
        self.sample_pairs = self.train_pairs

        # Load based on what to output
        self.transform = dataset_transform(self.phase, self.norm_family)
        self.loader = ImageLoader(ospj(self.root, 'images'))
        if self.use_precomputed_features:
            if self.args.use_gfn:
                with torch.no_grad():
                    feats_file = ospj(root, self.phase + '-' + model + '+' + args.gfn_arch + '_vectors.t7')
                    if not os.path.exists(feats_file):
                        self.activations, self.g_activations = self.generate_features(out_file=feats_file, model=model,
                                                                                      args=args)
                    else:
                        activation_data = torch.load(feats_file)
                        self.activations = dict(zip(activation_data['files'], activation_data['features']))
                        self.g_activations = dict(zip(activation_data['files'], activation_data['g_features']))
            else:
                with torch.no_grad():
                    feats_file = ospj(root, self.phase + '-' + model + '_feats_vectors.t7')
                    if not os.path.exists(feats_file):
                        self.activations, _ = self.generate_features(out_file=feats_file, model=model, args=args)
                    else:
                        activation_data = torch.load(feats_file)
                        self.activations = dict(zip(activation_data['files'], activation_data['features']))

    def generate_features(self, model, args, out_file=None):

        if self.phase == 'train':
            data = ospj(self.root, 'images')
            files_before = glob(ospj(data, '**', '*.jpg'), recursive=True)
            files_all = []
            for current in files_before:
                parts = current.split('/')
                if "cgqa" in self.root:
                    files_all.append(parts[-1])
                else:
                    files_all.append(os.path.join(parts[-2], parts[-1]))
        else:
            data = self.data
            files_all = []
            for item in data:
                files_all.append(item[0])
        transform = dataset_transform('all', self.norm_family)
        if args.use_gfn:
            gfn = get_image_extractor(arch=args.gfn_arch)
            gfn = gfn.eval().to(device)
            feat_extractor = get_image_extractor(arch=model)
            feat_extractor = feat_extractor.eval().to(device)

            image_feats = []
            gfn_feats = []
            image_files = []
            for chunk in tqdm(
                    chunks(files_all, 512), total=len(files_all) // 512, desc=f'Extracting features {model}'):
                files = chunk
                imgs = list(map(self.loader, files))
                imgs = list(map(transform, imgs))
                feats = feat_extractor(torch.stack(imgs, 0).to(device))
                g_feats = gfn(torch.stack(imgs, 0).to(device))

                image_feats.append(feats.data.cpu())
                gfn_feats.append(g_feats.data.cpu())
                image_files += files
            image_feats = torch.cat(image_feats, 0)
            gfn_feats = torch.cat(gfn_feats, 0)
            print('features for %d images generated' % (len(image_files)))
            activation = dict(zip(image_files, image_feats))
            g_activation = dict(zip(image_files, gfn_feats))
            torch.save({'features': image_feats, 'g_features': gfn_feats, 'files': image_files}, out_file)
            return activation, g_activation
        else:
            feat_extractor = get_image_extractor(arch=model).eval()
            feat_extractor = feat_extractor.to(device)
            image_feats = []
            image_files = []
            for chunk in tqdm(
                    chunks(files_all, 256), total=len(files_all) // 256, desc=f'Extracting features {model}'):
                files = chunk
                imgs = list(map(self.loader, files))
                imgs = list(map(transform, imgs))
                feats = feat_extractor(torch.stack(imgs, 0).to(device))
                image_feats.append(feats.data.cpu())
                image_files += files
            image_feats = torch.cat(image_feats, 0)
            print('features for %d images generated' % (len(image_files)))
            activation = dict(zip(image_files, image_feats))
            torch.save({'features': image_feats, 'files': image_files}, out_file)
            return activation, None

    def parse_split(self):

        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [line.split() for line in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            ospj(self.root, self.split, 'train_pairs.txt')
        )
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            ospj(self.root, self.split, 'val_pairs.txt')
        )
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            ospj(self.root, self.split, 'test_pairs.txt')
        )

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
            list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def get_split_info(self):
        data = torch.load(ospj(self.root, 'metadata_{}.t7'.format(self.split)))

        train_data, val_data, test_data = [], [], []

        for instance in data:
            image, attr, obj, settype = instance['image'], instance['attr'], \
                instance['obj'], instance['set']
            curr_data = [image, attr, obj]

            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                continue

            if settype == 'train':
                train_data.append(curr_data)
            elif settype == 'val':
                val_data.append(curr_data)
            else:
                test_data.append(curr_data)

        return train_data, val_data, test_data

    def get_dict_data(self, data, pairs):
        data_dict = {}
        for current in pairs:
            data_dict[current] = []

        for current in data:
            image, attr, obj = current
            data_dict[(attr, obj)].append(image)

        return data_dict

    def __getitem__(self, index):

        index = self.sample_indices[index]
        image, attr, obj = self.data[index]  # return image_path,attr name,obj name
        # Decide what to output，when phase is train,self.finetune_backbone is true
        if self.use_precomputed_features:
            img = self.activations[image]
            if self.args.use_gfn:
                g_img = self.g_activations[image]
        else:
            img = self.loader(image)
            img = self.transform(img)
            if self.args.use_gfn:
                g_img = img
        if self.args.use_gfn:
            data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)], g_img]
        else:
            data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]
        # Return image paths if requested as the last element of the list
        if self.return_images and self.phase != 'train':
            data.append(image)
        return data

    def __len__(self):

        return len(self.sample_indices)  # when train：30338

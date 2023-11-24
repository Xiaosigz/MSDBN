import logging
from PIL import Image
import os

import torch

from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from utils.dataset import CUB, CarsDataset, NABirds, dogs, INat2017
from utils.autoaugment import AutoAugImageNetPolicy
from torchvision import datasets
import torchvision.transforms as transforms
from utils.dataloader import config1, Dataset, collate_fn
logger = logging.getLogger(__name__)
# train_root, test_root, train_pd, cls_num = config(data='bird')


train_root, test_root, train_pd, test_pd, cls_num = config1('shark')
def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'shark':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         trainset = datasets.ImageFolder(root='imbASLO/train100',  transform=train_transform)
#         testset = datasets.ImageFolder(root='imbASLO/testing', transform = test_transform)
        train_dataset = Dataset(train_root, train_pd, train=True, transform=train_transform, num_positive=1)
#         test_dataset = Dataset(test_root, test_pd, train=False, transform=test_transform)
        test_dataset = Dataset(test_root, test_pd, train=False, transform=test_transform)
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
                            os.path.join(args.data_root,'cars_train'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
        testset = CarsDataset(os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
                            os.path.join(args.data_root,'cars_test'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transforms.Compose([
                                    transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
    elif args.dataset == 'dog':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root,
                                train=True,
                                cropped=False,
                                transform=train_transform,
                                download=False
                                )
        testset = dogs(root=args.data_root,
                                train=False,
                                cropped=False,
                                transform=test_transform,
                                download=False
                                )
    elif args.dataset == 'nabirds':
        train_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                        transforms.RandomCrop((448, 448)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
                                        transforms.CenterCrop((448, 448)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    elif args.dataset == 'INat2017':
        train_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.RandomCrop((304, 304)),
                                    transforms.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
                                    transforms.CenterCrop((304, 304)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(args.data_root, 'train', train_transform)
        testset = INat2017(args.data_root, 'val', test_transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
#     test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)
    return train_loader, test_loader

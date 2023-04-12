import os
import random
import sys
import pickle
import program_
import argparse
import matplotlib.pyplot as plt
import numpy as np
from random import randrange
import queue
import math
import torch
from torchvision import datasets, transforms
from evolution_program import *
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from resnet_cifar import resnet18
from googlenet import googlenet, GoogLeNet
from densenet import densenet121, densenet161, densenet169
import pandas as pd
import torch.multiprocessing as tmp
from torchvision.models import resnet50, efficientnet_b0, densenet121
from synthesize import run_program
from utils_ import *

if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devices = []
    num_gpus = torch.cuda.device_count()
    if torch.cuda.is_available():
        print('Using', num_gpus, "GPUs")
        for i in range(num_gpus):
            devices.append(torch.device('cuda:' + str(i)))

    parser = argparse.ArgumentParser(description='OPPSLA synthesizer')
    parser.add_argument('--model', default='vgg16', type=str, help='model')
    parser.add_argument('--data_set', default='cifar10', type=str, help='data set - must be CIFAR-10 or ImageNet')
    parser.add_argument('--classes_list', metavar='N', type=int, nargs='+', help='classes for the synthesis process')
    parser.add_argument('--imagenet_dir', type=str, help='directory for images of ImageNet dataset')
    parser.add_argument('--program_path', type=str, help='path of the program as a pkl file')
    parser.add_argument('--results_path', default="./results_OPPSLA", type=str, help='path of the saved results')
    parser.add_argument('--g', default=0, type=int, help='level of granularity')
    parser.add_argument('--max_g', default=0, type=int, help='number of pixels with finer granularity')
    
    args = parser.parse_args()


    if args.data_set == "cifar10":
        img_dim = 32
        transform = transforms.Compose(
            [transforms.ToTensor()])
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                  download=True, transform=transform)
    elif args.data_set == "imagenet":
        img_dim = 224
        if args.imagenet_dir is None:
            raise Exception("imagenet_dir must be not None")
        if not os.path.exists('./' + args.imagenet_dir):
            raise Exception("can't find the directory for ImageNet")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        test_data = datasets.ImageFolder(
            os.path.join(args.imagenet_dir),
            transforms.Compose([
                transforms.Resize(img_dim),
                transforms.CenterCrop(img_dim),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]))


    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=1)
    if args.model == "vgg16":
        model = vgg16_bn()
        model.load_state_dict(torch.load("vgg16_bn.pt", map_location='cpu'))
    elif args.model == "resnet18":
        model = resnet18()
        model.load_state_dict(torch.load("resnet18.pt", map_location='cpu'))
    elif args.model == "GoogLeNet":
        model = GoogLeNet()
        model.load_state_dict(torch.load("googlenet.pt", map_location='cpu'))

    model = model.to(device)
    model.eval()
    program_dict = pickle.load(open(args.program_path, 'rb'))
    program = program_dict[0]
    center_matrix = generate_center_matrix(img_dim, device)

    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
    num_classes = len(args.classes_list)
    args.classes_list = [args.classes_list[i:i + num_gpus] for i in range(0, len(args.classes_list), num_gpus)]
    ctx = tmp.get_context('spawn')
    with tqdm(total=num_classes, desc="Attacking",
              bar_format="{l_bar}{bar:10}{r_bar}") as pbar:
        for i in range(len(args.classes_list)):
            processes = []
            for j in range(len(args.classes_list[i])):
                processes.append(ctx.Process(target=run_program, \
                                             args=(program, model, test_loader, img_dim, center_matrix, args.max_g,\
                                                   args.g, devices[j], True, args.classes_list[i][j], args.results_path)))
            for proc in processes:
                proc.start()
            for proc in processes:
                proc.join()
            pbar.update(len(args.classes_list[i]))

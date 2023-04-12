import os
import random
import sys
import pickle
import program_
import argparse
import numpy as np
from random import randrange
import queue
import math
import torch
from torchvision import datasets, transforms
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
from torch.utils.data import DataLoader
from torchvision.models import resnet50, densenet121
from MH import run_MH
from utils_ import *


# def select_n_images_imagenet(n, true_label, data_loader, model, values_dict, k_user, device):
#     good_idx_list = []
#     good_class_list = []
#     for batch_idx, (data, target) in enumerate(data_loader):
#         is_success = False
#         img_x, img_y = data.to(device), target.to(device)
#         if img_y.item() not in [16, 17]:
#             continue
#         softmax = nn.Softmax(dim=1)
#         predictions_vector = softmax(model(img_x).data)
#         pred = torch.argmax(predictions_vector)
#         if pred.item() != img_y.item():
#             continue
#         print(batch_idx, "$$$", img_y)
#         select_n_queries = 0
#         possible_directions_list = create_sorted_loc_pert_list(img_x, values_dict)
#         for coor in possible_directions_list:
#             if is_success:
#                 break
#             select_n_queries += 1
#             # if select_n_queries > 150000:
#             #     break
#             x, y = coor[0]
#             pert_type = coor[1]
#             is_success, queries, curr_prob = try_perturb_pixel(x, y, model, img_x, img_y, pert_type, device)
#         if is_success:
#             if img_y.item() not in good_class_list or True:
#                 good_idx_list.append(batch_idx)
#                 good_class_list.append(img_y.item())
#                 print("SUCCESS!!")
#             if len(good_idx_list) == n:
#                 return good_idx_list
#     return good_idx_list


def run_program(program, model, dataloader, img_dim, center_matrix, max_g, g, device, is_test=False, class_idx=None, results_path=None):
    """
    Run the specified program for adversarial attacks on a given model using the provided dataloader.

    Args:
        program (Program): A Program object containing the conditions for the adversarial attack.
        model (nn.Module): The neural network model to be attacked.
        dataloader (DataLoader): DataLoader containing the input images and labels.
        img_dim (int): The dimension of the input image (assuming a square image).
        center_matrix (torch.Tensor): A matrix representing the distance of each pixel to the image center.
        max_g (int): The maximum number of pixels to perturb with finer granularity.
        g (int): The level of granularity.
        device (torch.device): The device to perform computations on (e.g., 'cuda' or 'cpu').
        is_test (bool, optional): Whether the current run is for the test set. Defaults to False.
        class_idx (int, optional): Index of the current class. Required if is_test is True. Defaults to None.
        results_path (str, optional): Path to the directory where the results CSV file should be saved.

    Returns:
        float: The average number of queries required to succeed in the adversarial attack.
    """
    model = model.to(device)
    model.eval()
    pert_type_to_idx_dict = create_pert_type_to_idx_dict()
    if is_test:
        results_df = pd.DataFrame(columns=["batch_idx", "class", "is_success", "queries"])
    num_imgs, num_success, sum_queries = 0, 0, 0
    max_queries = 1000000000
    for batch_idx, (data, target) in enumerate(dataloader):
        is_success = False
        img_x, img_y = data.to(device), target.to(device)
        if is_test:
            if img_y.item() != class_idx:
                continue
        if not is_correct_prediction(model, img_x, img_y):
            continue
        num_imgs += 1
        possible_loc_pert_list = create_sorted_loc_pert_list(img_x)
        possible_loc_pert_list.append("STOP")
        indicators_tensor = torch.zeros((8, img_dim, img_dim))
        orig_prob = get_orig_confidence(model, img_x, img_y, device)
        n_queries = 0
        min_prob_dict = {}
        for loc_pert in possible_loc_pert_list:
            if n_queries >= max_queries or is_success:
                break
            if loc_pert == "STOP":
                sorted_loc_list = sorted(min_prob_dict.items(), key=lambda x: x[1])
                for loc_idx in range(max_g):
                    if g <= 0:
                        break
                    is_success, queries, curr_prob = try_perturb_pixel_finer_granularity(sorted_loc_list[loc_idx][0][0],
                                                                                         sorted_loc_list[loc_idx][0][1],
                                                                                         model, img_x, img_y, g, device)
                    n_queries += queries
                    if is_success:
                        sum_queries += n_queries
                continue

            x, y = loc_pert[0]
            pert_type = loc_pert[1]
            if indicators_tensor[pert_type_to_idx_dict[pert_type]][x][y] > 0:
                continue
            is_success, queries, curr_prob = try_perturb_pixel(x, y, model, img_x, img_y, pert_type, device)
            indicators_tensor[pert_type_to_idx_dict[pert_type]][x][y] = 1
            n_queries += queries
            if is_success:
                sum_queries += n_queries
                break
            update_min_confidence_dict(min_prob_dict, x, y, curr_prob)
            if check_cond(program.cond_1, img_x, x, y, orig_prob, curr_prob, center_matrix):
                for r in [-1, 0, 1]:
                    for s in [-1, 0, 1]:
                        new_x = max(min(img_dim - 1, x + r), 0)
                        new_y = max(min(img_dim - 1, y + s), 0)
                        if indicators_tensor[pert_type_to_idx_dict[pert_type]][new_x][new_y] == 0:
                            possible_loc_pert_list.append(possible_loc_pert_list. \
                                pop(possible_loc_pert_list.index(((new_x, new_y), pert_type))))
            if check_cond(program.cond_2, img_x, x, y, orig_prob, curr_prob, center_matrix):
                new_pert_type = next((elem[1] for elem in possible_loc_pert_list \
                                      if (elem[0] == (x, y) and pert_type != elem[1])), None)
                if new_pert_type is not None:
                    if indicators_tensor[pert_type_to_idx_dict[new_pert_type]][x][y] == 0:
                        possible_loc_pert_list.append(possible_loc_pert_list. \
                            pop(possible_loc_pert_list.index(((x, y), new_pert_type))))

            pixels_probs_list_wide, pixels_probs_list_deep = initialize_pixels_conf_lists(x, y, pert_type, curr_prob)
            while ((not pixels_probs_list_wide.empty()) or (not pixels_probs_list_deep.empty())) and not is_success:
                if n_queries >= max_queries:
                    break
                while (not pixels_probs_list_wide.empty()) and (not is_success):
                    pixel_prob = pixels_probs_list_wide.get()
                    best_x, best_y = pixel_prob[0]
                    pert_type = pixel_prob[1]
                    if check_cond(program.cond_3, img_x, best_x, best_y, orig_prob, pixel_prob[2], center_matrix):
                        for r in [-1, 0, 1]:
                            for s in [-1, 0, 1]:
                                if is_success or n_queries >= max_queries:
                                    break
                                new_x = max(min(img_dim - 1, x + r), 0)
                                new_y = max(min(img_dim - 1, y + s), 0)
                                if indicators_tensor[pert_type_to_idx_dict[pert_type]][new_x][new_y] > 0:
                                    continue
                                is_success, queries, curr_prob = try_perturb_pixel(new_x, new_y, model,
                                                                                   img_x, img_y, pert_type, device)
                                indicators_tensor[pert_type_to_idx_dict[pert_type]][new_x][new_y] = 1
                                n_queries += queries
                                update_min_confidence_dict(min_prob_dict, new_x, new_y, curr_prob)
                                pixels_probs_list_wide.put(((new_x, new_y), pert_type, curr_prob))
                                pixels_probs_list_deep.put(((new_x, new_y), pert_type, curr_prob))
                                if is_success:
                                    sum_queries += n_queries
                while (not pixels_probs_list_deep.empty()) and (not is_success):
                    pixel_prob = pixels_probs_list_deep.get()
                    new_x, new_y = pixel_prob[0]
                    if check_cond(program.cond_4, img_x, new_x, new_y, orig_prob, pixel_prob[2], center_matrix):
                        pert_type = pixel_prob[1]
                        new_pert_type = next((elem[1] for elem in possible_loc_pert_list \
                                              if (elem[0] == (new_x, new_y) and pert_type != elem[1])), None)
                        if new_pert_type is not None:
                            if indicators_tensor[pert_type_to_idx_dict[new_pert_type]][new_x][new_y] > 0:
                                continue
                            if n_queries >= max_queries:
                                break
                            is_success, queries, curr_prob = try_perturb_pixel(new_x, new_y, model,
                                                                               img_x, img_y, new_pert_type, device)
                            indicators_tensor[pert_type_to_idx_dict[new_pert_type]][new_x][new_y] = 1
                            update_min_confidence_dict(min_prob_dict, new_x, new_y, curr_prob)
                            n_queries += queries
                            pixels_probs_list_wide.put(((new_x, new_y), new_pert_type, curr_prob))
                            pixels_probs_list_deep.put(((new_x, new_y), new_pert_type, curr_prob))
                            if is_success:
                                sum_queries += n_queries
                                break

        if is_success:
            num_success += 1

        if is_test:
            results_df = update_results_df(results_df, results_path, batch_idx, class_idx, is_success, n_queries)

    return sum_queries / num_success


def synthesize(args):
    """
    Synthesizes a program for each specified class using the given arguments.

    Args:
        args (argparse.Namespace): An object containing the parsed command-line arguments, including:
            - args.model (str): The name of the model to use.
            - args.data_set (str): The name of the dataset to use.
            - args.classes_list (list[int]): A list of class indices to synthesize programs for.
            - args.num_train_images (int): The number of images in the training set.
            - args.max_iter (int): The maximum number of iterations for the MH algorithm.
            - args.num_iter_stop (int): The number of iterations without change before stopping.
            - args.g (int): The level of granularity.
            - args.max_g (int): The number of pixels with finer granularity.

    Returns:
        None

    Side Effects:
        - Saves the synthesized program for each class to a '.pkl' file.
        - Writes the synthesis results to a '.txt' file.
        - Prints the progress of the synthesis process.
    """
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    devices = []
    num_gpus = torch.cuda.device_count()
    if torch.cuda.is_available():
        print('Using', num_gpus, "GPUs")
        for i in range(num_gpus):
            devices.append(torch.device('cuda:' + str(i)))

    train_data, img_dim = get_data_set(args)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=1)
    model = load_model(args.model, device)
    center_matrix = generate_center_matrix(img_dim, device)

    with open(args.model + '_' + args.data_set + '.txt', 'w') as f:
        f.write("data set: " + args.data_set + "\n")
        f.write("number of training images: " + str(args.num_train_images) + "\n")

    program_dict = {}

    for class_idx in args.classes_list:
        print("########################")
        print("synthesizing program for class : ", class_idx)
        print("########################")
        train_imgs_idx = select_n_images(args.num_train_images, class_idx, train_loader, model, args.max_g, args.g, device)
        n_train_data = torch.utils.data.Subset(train_data, train_imgs_idx)
        data_loader = torch.utils.data.DataLoader(n_train_data, shuffle=False, batch_size=1)
        best_program = program_.Program(img_dim)
        best_queries = run_program(best_program, model, data_loader, img_dim, center_matrix, args.max_g, args.g, device)
        previous_best_queries = None
        num_same_best_queries_iter = 1
        with tqdm(total=args.max_iter, desc="Synthesizing program",
                  bar_format="{l_bar}{bar:10}{r_bar}") as pbar:

            for iter_idx in range(int(args.max_iter / 10)):
                if previous_best_queries == best_queries:
                    num_same_best_queries_iter += 1
                else:
                    num_same_best_queries_iter = 1
                if num_same_best_queries_iter == int(args.num_iter_stop / 10):
                    break
                previous_best_queries = best_queries
                queue_proc = tmp.Manager().Queue()
                ctx = tmp.get_context('spawn')
                processes = []
                for device_ in devices:
                    processes.append(ctx.Process(target=run_MH, \
                        args=(best_program, best_queries, model, data_loader, img_dim, center_matrix, 10, queue_proc, args.max_g,\
                              args.g, device_)))

                for proc in processes:
                    proc.start()
                for proc in processes:
                    proc.join()
                best_program, best_queries = update_best_program(queue_proc, best_program, best_queries)
                pbar.update(10)
            program_dict[class_idx] = best_program
            pickle.dump(program_dict, open(args.model + '_' + args.data_set + '.pkl', 'wb'))
            write_program_results(args, class_idx, best_program, best_queries)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OPPSLA Synthesizer')
    parser.add_argument('--model', default='vgg16', type=str,
                        help='Model architecture to use (e.g., vgg16, resnet18, etc.)')
    parser.add_argument('--data_set', default='cifar10', type=str, help='Dataset to use - must be CIFAR-10 or ImageNet')
    parser.add_argument('--classes_list', metavar='N', type=int, nargs='+', help='List of classes for the synthesis')
    parser.add_argument('--num_train_images', default=50, type=int, help='# of images in the training set per class')
    parser.add_argument('--imagenet_dir', type=str, help='Directory containing ImageNet dataset images')
    parser.add_argument('--max_iter', default=210, type=int, help='Maximum # of iterations for the MH algorithm')
    parser.add_argument('--num_iter_stop', default=60, type=int, help='# of iterations without change before stopping the algorithm')
    parser.add_argument('--g', default=0, type=int, help='Granularity level for the synthesis process')
    parser.add_argument('--max_g', default=0, type=int, help='Maximum number of pixels with finer granularity')

    args = parser.parse_args()
    synthesize(args)


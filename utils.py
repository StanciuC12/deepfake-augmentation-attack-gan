import argparse
import os
import numpy as np
import math
import sys
import csv
import random
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from PIL import Image
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples):

    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

def replace_bn_with_ln(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.LayerNorm(child.num_features))
        else:
            replace_bn_with_ln(child)


def calc_gradient_penalty(Discriminator, real, fake):
    m = real.shape[0]
    epsilon = torch.rand(m, 1, 1, 1)
    if cuda:
        epsilon = epsilon.cuda()

    interpolated_img = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)
    interpolated_out = Discriminator(interpolated_img)

    grads = autograd.grad(outputs=interpolated_out, inputs=interpolated_img,
                          grad_outputs=torch.ones(interpolated_out.shape).cuda() if cuda else torch.ones(
                              interpolated_out.shape),
                          create_graph=True, retain_graph=True)[0]
    grads = grads.reshape([m, -1])
    gradients_norm = torch.sqrt(torch.sum(grads ** 2, dim=1) + 1e-12)
    #grad_penalty = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    grad_penalty = ((gradients_norm - 1) ** 2).mean()
    return grad_penalty



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


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def save_result_image(output_tensor, save_path,
                      mean=torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32),
                      std=torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)):

    # Define the inverse normalization transform
    inverse_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    # Inverse transform the output tensor
    output_inverse = inverse_normalize(output_tensor)

    # Convert the tensor to a NumPy array
    output_numpy = output_inverse.cpu().detach().numpy()

    # Rescale pixel values to the valid range [0, 255]
    #output_numpy = ((output_numpy + 1) / 2.0) * 255.0
    output_numpy[output_numpy < 0] = 0
    output_numpy[output_numpy > 1] = 1
    output_numpy = output_numpy * 255.0
    # Convert to uint8 data type
    output_numpy = output_numpy.astype(np.uint8)

    # Convert the NumPy array to a PIL Image
    output_image = Image.fromarray(output_numpy.transpose(1, 2, 0))

    # Save the resulting image
    output_image.save(save_path)


# def select_random_images_multiple_folders(folder_path, num_images=4, image_type='train', ignore_folder=None):
#
#     # Get a list of subfolders
#     subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder)) and subfolder != ignore_folder]
#
#     # Select one subfolder at random
#     selected_subfolder = random.choice(subfolders)
#
#     # Get a list of image files in the selected subfolder
#     subfolder_path = os.path.join(folder_path, selected_subfolder)
#     image_files = [file for file in os.listdir(subfolder_path) if '.png' in file and image_type in file]
#
#     # Select num_images random image addresses
#     random_image_addresses = random.sample([os.path.join(subfolder_path, image) for image in image_files], num_images)
#
#     return random_image_addresses

def select_random_images_multiple_folders(folder_path, num_images=4, image_type='train', ignore_folder=None):

    # Get a list of subfolders
    subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder)) and subfolder != ignore_folder]

    # Select num_images random subfolders
    selected_subfolders = random.choices(subfolders, k=num_images)

    random_image_addresses = []

    for selected_subfolder in selected_subfolders:
        # Get a list of image files in the selected subfolder
        subfolder_path = os.path.join(folder_path, selected_subfolder)
        image_files = [file for file in os.listdir(subfolder_path) if '.png' in file]

        if image_files:
            # Select one random image address from the selected subfolder
            random_image_address = os.path.join(subfolder_path, random.choice(image_files))
            random_image_addresses.append(random_image_address)

    return random_image_addresses


def select_random_images_folder(folder_path, num_images=4, image_type='train'):

    image_files = [file for file in os.listdir(folder_path) if '.png' in file and image_type in file]
    # Select num_images random image addresses
    random_image_addresses = random.sample([os.path.join(folder_path, image) for image in image_files], num_images)

    return random_image_addresses

def load_images_from_adr(img_adresses, labels, transform):

    i = 0
    data = []
    for img in img_adresses:

        X = np.array(Image.open(img), dtype=np.float32)
        if np.max(X) > 1:
            X = X / 255

        X = transform(X)
        if X is not None:
            data.append(X)

        i += 1

    labels = torch.Tensor(labels)
    data = torch.stack(data)

    return data, labels



"""
Utilities of Project
"""
import numpy as np
import argparse
import torch
import os
import torch.nn.utils as utils
from yaml import parse
import torch.nn as nn
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def get_args():
    """
    The argument that we have defined, will be used in training and evaluation(infrence) modes
    """
    parser = argparse.ArgumentParser(
        description="Arguemnt Parser of `Train` and `Evaluation` of our network"
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        default=2,
        type=int,
        help="Number of data in each batch",
    )

    parser.add_argument(
        "--lr", dest="lr", default=1e-3, type=float, help="Learning rate value"
    )

    parser.add_argument(
        "--momentum",
        dest="momentum",
        default=0.9,
        type=float,
        help="Momentum coefficient",
    )

    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        default=4e-6,
        type=float,
        help="Weight decay value",
    )

    parser.add_argument(
        "--num-epochs",
        dest="num_epochs",
        default=25,
        type=int,
        help="Number of epochs",
    )

    parser.add_argument(
        "--gpu", dest="gpu", default=True, type=bool, help="wheather to use gpu or not"
    )

    parser.add_argument(
        "--ckpt-save-path",
        dest="ckpt_save_path",
        default="../ckpts",
        type=str,
        help="base path(folder) to save model ckpts",
    )

    parser.add_argument(
        "--ckpt-prefix",
        dest="ckpt_prefix",
        default="cktp_epoch_",
        type=str,
        help="prefix name of ckpt which you want to save",
    )

    parser.add_argument(
        "--ckpt-save-freq",
        dest="ckpt_save_freq",
        default=10,
        type=int,
        help="after how many epoch(s) save model",
    )

    parser.add_argument(
        "--ckpt-load-path",
        dest="ckpt_load_path",
        type=str,
        default=None,
        help="Checkpoints address for loading",
    )

    parser.add_argument(
        "--report-path",
        dest="report_path",
        type=str,
        default="../reports",
        help="Saving report directory",
    )

    parser.add_argument(
        "--x-path",
        dest="x_path",
        type=str,
        default="./data/trainSet/Stimuli",
        help="Path of images(x)",
    )

    parser.add_argument(
        "--y-path",
        dest="y_path",
        type=str,
        default="./data/trainSet/FIXATIONMAPS",
        help="Path of Fixationmaps(y)",
    )

    parser.add_argument(
        "--regex-for-category",
        dest="regex_for_category",
        type=str,
        default="\.\/data\/trainSet\/Stimuli\/(.*)\/\d*\.jpg",
        help="Regex of images(x) which will be used for category",
    )

    options = parser.parse_args()

    return options


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved.Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save

    """
    state_dict = dict()

    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])

    return model, optimizer


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group["params"], max_norm, norm_type)


def show(input, bgr=False):

    if bgr == False:
        image_number_for_plot = input.shape[0]
        fig, ax = plt.subplots(1, image_number_for_plot, figsize=(12, 4))
        for index in range(image_number_for_plot):
            curr_image = input[index, :, :, :].detach().numpy().transpose((1, 2, 0))
            mean = np.array(0.5)
            std = np.array(0.5)
            curr_image = std * curr_image + mean
            curr_image = np.clip(curr_image, 0, 1)
            ax[index].imshow(curr_image[:, :, 0])

    if bgr == True:
        image_number_for_plot = input.shape[0]
        fig, ax = plt.subplots(1, image_number_for_plot, figsize=(12, 4))
        for index in range(image_number_for_plot):
            curr_image = input[index, :, :, :].detach().numpy().transpose((1, 2, 0))
            mean = np.array(0.5)
            std = np.array(0.5)
            curr_image = std * curr_image + mean
            curr_image = np.clip(curr_image, 0, 1)
            ax[index].imshow(curr_image[:, :, [2, 1, 0]])
    plt.tight_layout()
    plt.show()


def loss_optimizer(
    net,
    lr=10e-4,
    momentum=0.9,
    nesterov=False,
    weight_decay=4e-6,
    step_size=0.5,
    gamma=0.5,
):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=momentum,
        nesterov=nesterov,
        weight_decay=weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    return [criterion, optimizer, scheduler]


def plot_samples(net, myDataLoader, num_of_repeating_loader=3, batch_size=2):

    num_of_rows = num_of_repeating_loader * batch_size
    cols_name = ["Original", "Saliency", "Predicted Saliency"]

    mean_image = np.array([0.5, 0.5, 0.5])
    std_image = np.array([0.5, 0.5, 0.5])

    mean_output = np.array([0.5])
    std_output = np.array([0.5])

    fig, axes = plt.subplots(num_of_rows, 3, figsize=(20, 30))

    plt.subplots_adjust(wspace=0.05, hspace=0)
    for ax, col in zip(axes[0], cols_name):
        ax.set_title(col)

    image_index = 0
    for num_of_repeating_loader_ in range(num_of_repeating_loader):
        real_images, saliencies = next(iter(myDataLoader))
        net.to("cpu")
        predicteds = net(real_images)
        for batch_number in range(batch_size):

            real_image = (
                real_images[batch_number, :, :, :].detach().numpy().transpose((1, 2, 0))
            )
            real_image = std_image * real_image + mean_image
            real_image = np.clip(real_image, 0, 1)
            axes[image_index, 0].imshow(real_image[:, :, [2, 1, 0]])
            axes[image_index, 0].set_axis_off()

            saliency = (
                saliencies[batch_number, :, :, :].detach().numpy().transpose((1, 2, 0))
            )
            saliency = std_output * saliency + mean_output
            saliency = np.clip(saliency, 0, 1)
            axes[image_index, 1].imshow(saliency)
            axes[image_index, 1].set_axis_off()

            predicted = (
                predicteds[batch_number, :, :, :].detach().numpy().transpose((1, 2, 0))
            )
            predicted = std_output * predicted + mean_output
            predicted = np.clip(predicted, 0, 1)
            axes[image_index, 2].imshow(predicted)
            axes[image_index, 2].set_axis_off()
            image_index += 1
    plt.show()


def plot_loss(list_of_train_loss, list_of_test_loss, epochs, fig_size=(20, 10)):

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0, 0, 1, 1])

    for label in ax.xaxis.get_ticklabels():

        label.set_color("black")
        label.set_rotation(0)
        label.set_fontsize(10)

    ax.xaxis.set_major_locator(plt.MaxNLocator(epochs))
    ax.plot(
        [str(epoch) for epoch in list(range(1, len(list_of_train_loss) + 1))],
        list_of_train_loss,
        color="b",
        label="train_loss",
    )

    ax.plot(
        [str(epoch) for epoch in list(range(1, len(list_of_test_loss) + 1))],
        list_of_test_loss,
        color="r",
        label="test_loss",
    )

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(loc=0)

import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn

######  Local packages  ######
import learning as learning
import models as models
import data as data
import util as util


def main(args):
    cuda = True if args.gpu and torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    train_path, val_path = data.train_val_pathes(
        args.x_path, args.y_path, args.regex_for_category
    )
    transform_image, transform_mask = data.transformer()

    train_dataloader, val_dataloader = data.train_val_loader(
        train_path, val_path, transform_image, transform_mask, args.batch_size
    )

    model = models.Net()

    criterion = nn.MSELoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Loading Model
    if args.ckpt_load_path is not None:
        print("******  Loading Model   ******")
        model, optimizer = util.load_model(
            ckpt_path=args.ckpt_load_path, model=model, optimizer=optimizer
        )

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Schedular
    num_train_steps = (
        math.ceil(len(train_dataloader) / args.batch_size) * args.num_epochs
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # Train the model(Train and Validation Steps)
    model, optimizer = learning.Train_mode(
        model=model,
        cuda=cuda,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=args.num_epochs,
        saving_checkpoint_path=args.ckpt_save_path,
        saving_prefix=args.ckpt_prefix,
        saving_checkpoint_freq=args.ckpt_save_freq,
        report_path=args.report_path,
    )

    return model


if __name__ == "__main__":
    args = util.get_args()
    main(args)

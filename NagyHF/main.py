import math
import os
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import jaccard
from torch.utils.data import DataLoader

from GraphConvModel import GraphConvModel
from crn import RefinementNetwork
from graph import GraphTripleConvNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision.transforms as T

from prepareData.prepareData import coco_collate_fn, build_coco_dsets

# Calculate the loss after each iteration, the loss functions contains the errors of the masks and bounding boxes.
def calculate_model_losses(skip_pixel_loss, img, img_pred, bbox, bbox_pred, masks, masks_pred):
    total_loss = torch.zeros(1).to(img)
    losses = {}

    l1_pixel_weight = 1
    bbox_pred_loss_weight = 10
    mask_loss_weight = 0.1

    if skip_pixel_loss:
        l1_pixel_weight = 0

    l1_pixel_loss = F.l1_loss(img_pred, img)
    total_loss = add_loss(total_loss, l1_pixel_loss, losses, 'L1_pixel_loss', l1_pixel_weight)

    loss_bbox = F.mse_loss(bbox_pred, bbox)
    total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred', bbox_pred_loss_weight)

    mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
    total_loss = add_loss(total_loss, mask_loss, losses, 'mask_loss', mask_loss_weight)

    return total_loss, losses

# Helper function for calculating total loss
def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss

# The main function that trains the GraphConvModel's network
def train():
    checkpoint = {
        'losses_ts': [],
        'losses': defaultdict(list)
    }

    # for using our GPU
    float_dtype = torch.cuda.FloatTensor

    # Building the dataset
    # the vocab contains the vocabulary of the scenegraphs
    vocab, train_dset = build_coco_dsets()

    # parameters of the DataLoader.
    #coco_collate_fn is a funtion that gets the batch and returns the images, objects, boxes, masks, triples, obj_to_img, triple_to_img
    loader_kwargs = {
        'batch_size': 32,
        'num_workers': 4,
        'shuffle': True,
        'collate_fn': coco_collate_fn,
    }


    # Loading data and grouping the input data to batches, during the epoch we can iterate through the whole dataset.
    # We are using torch.utils.data DataLoader function
    train_loader = DataLoader(train_dset, **loader_kwargs)

    # Loading the model
    model = GraphConvModel(vocab)

    model.to(torch.device("cuda:0"))
    model.type(float_dtype)
    print(model)


    #We are using Adam optimizer with 0.0001 learing rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    iteration, epoch = 0, 0

    #We are training the model trought 100000 iterations
    while iteration < 100000:

        print('Starting epoch %d' % epoch)

        epoch += 1

        for batch in train_loader:

            # after 10 000 iteration starting eval mode
            if iteration == 10000:
                print('switching to eval mode')
                model.eval()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            iteration += 1

            batch = [tensor.cuda() for tensor in batch]
            imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch

            model_boxes = boxes
            model_masks = masks

            #forward the batch data in the model
            model_out = model(objs, triples, obj_to_img, boxes_gt=model_boxes, masks_gt=model_masks)

            boxes_pred, masks_pred, imgs_pred = model_out

            #printing original and predicted images every 50 iterations
            if (iteration % 50 == 0):
                invTrans = T.Compose([T.Normalize(mean=[0., 0., 0.],
                                                  std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                      T.Normalize(mean=[-0.485, -0.456, -0.406],
                                                  std=[1., 1., 1.]),
                                      ])

                image = invTrans(imgs[0])
                imagePred = invTrans(imgs_pred[0])
                plt.figure()
                plt.imshow(imagePred.permute(1, 2, 0).cpu().detach().numpy(), interpolation='nearest')
                plt.figure()
                plt.imshow(image.permute(1, 2, 0).cpu().detach().numpy(), interpolation='nearest')
                plt.show()

            #calculating the loss after the prediction
            skip_pixel_loss = (model_boxes is None)
            total_loss, losses = calculate_model_losses(skip_pixel_loss, imgs, imgs_pred, boxes, boxes_pred, masks,
                                                        masks_pred)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue

            #Starting backpropagation based on the total loss of the model and stepping the optimizer
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            #saving the model state into the model.pt file and printing the losses
            if iteration % 100 == 0:
                print('t = %d / %d' % (iteration, 100000))
                for name, val in losses.items():
                    print(' G [%s]: %.4f' % (name, val))
                    checkpoint['losses'][name].append(val)
                checkpoint['losses_ts'].append(iteration)

                saveModel = {}
                saveModel['model'] = model.state_dict()
                checkpoint_path = os.path.join(os.getcwd(), 'model.pt')
                torch.save(saveModel, checkpoint_path)
    return


if __name__ == '__main__':
    train()

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


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss = curr_loss * weight
    loss_dict[loss_name] = curr_loss.item()
    if total_loss is not None:
        total_loss += curr_loss
    else:
        total_loss = curr_loss
    return total_loss


def train():
    checkpoint = {
        'losses_ts': [],
        'losses': defaultdict(list)
    }

    float_dtype = torch.cuda.FloatTensor

    vocab, train_dset, val_dset = build_coco_dsets()

    loader_kwargs = {
        'batch_size': 32,
        'num_workers': 4,
        'shuffle': True,
        'collate_fn': coco_collate_fn,
    }

    train_loader = DataLoader(train_dset, **loader_kwargs)
    val_loader = DataLoader(val_dset, **loader_kwargs)

    model = GraphConvModel(vocab)
    model.to(torch.device("cuda:0"))
    model.type(float_dtype)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    iteration, epoch = 0, 0

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


            model_out = model(objs, triples, obj_to_img, boxes_gt=model_boxes, masks_gt=model_masks)

            boxes_pred, masks_pred, imgs_pred = model_out


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
                # plt.imshow(imgs_pred[0].permute(1, 2, 0).cpu().detach().numpy(), interpolation='nearest')
                # plt.figure()
                plt.imshow(image.permute(1, 2, 0).cpu().detach().numpy(), interpolation='nearest')
                plt.show()

            skip_pixel_loss = (model_boxes is None)

            total_loss, losses = calculate_model_losses(skip_pixel_loss, imgs, imgs_pred, boxes, boxes_pred, masks,
                                                        masks_pred)

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print('t = %d / %d' % (iteration, 100000))
                for name, val in losses.items():
                    print(' G [%s]: %.4f' % (name, val))
                    checkpoint['losses'][name].append(val)
                checkpoint['losses_ts'].append(iteration)

                saveModel = {}
                saveModel['model'] = model.state_dict()
                checkpoint_path = os.path.join(os.getcwd(), 'model')
                torch.save(saveModel, checkpoint_path)
    return


if __name__ == '__main__':
    '''
    args = {
        'batch_size': 32,
        'num_workers': 4,
        'shuffle': True,

        'image_size': (64, 64),
        'embedding_dim': 128,
        'gconv_dim': 128,
        'gconv_hidden_dim': 512,
        'gconv_num_layers': 5,
        'mlp_normalization': 'batch',
        'refinement_dims': (1024, 512, 256, 128, 64),
        'normalization': 'batch',
        'activation': 'relu',
        'mask_size': 16,
        'layout_noise_dim': 32,
        'num_iterations': 1000000,
        'learning_rate': 0.0001,
        'dataset': "coco",
        'num_train_samples': 8192,
        'num_val_samples': 1024,
        'timing': '1',

        'bbox_pred_loss_weight': 10,
        'mask_loss_weight': 0.1,
        'eval_mode_after': 100000,
        'print_every': 10,
        'checkpoint_every': 5,
        'output_dir': os.getcwd(),
        'checkpoint_name': 'checkpoint',
        'checkpoint_start_from': None,
        'restore_from_checkpoint': False,
        'l1_pixel_loss_weight': 1,
        'predicate_pred_loss_weight': 0,
    }
    '''
    train()

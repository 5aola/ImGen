# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import os
from collections import defaultdict
from timeit import timeit

import numpy as np
from scipy.spatial.distance import jaccard
from torch.utils.data import DataLoader

from GraphConvModel import GraphConvModel
from graph import GraphTripleConvNet
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepareData.prepareData import coco_collate_fn, build_coco_dsets


def calculate_model_losses(args, skip_pixel_loss, model,  img_pred,
                           bbox, bbox_pred, masks, masks_pred,
                           predicates):
  total_loss = torch.zeros(1).to(img_pred[0])
  losses = {}

  loss_bbox = F.mse_loss(bbox_pred, bbox)
  total_loss = add_loss(total_loss, loss_bbox, losses, 'bbox_pred',
                        args['bbox_pred_loss_weight'])

  if args['mask_loss_weight'] > 0 and masks is not None and masks_pred is not None:
    mask_loss = F.binary_cross_entropy(masks_pred, masks.float())
    total_loss = add_loss(total_loss, mask_loss, losses, 'mask_loss',
                          args['mask_loss_weight'])
  return total_loss, losses



def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
  curr_loss = curr_loss * weight
  loss_dict[loss_name] = curr_loss.item()
  if total_loss is not None:
    total_loss += curr_loss
  else:
    total_loss = curr_loss
  return total_loss

def build_loaders(args):

    print("In build_loaders")
    print(args)
    vocab, train_dset, val_dset = build_coco_dsets()
    collate_fn = coco_collate_fn

    loader_kwargs = {
        'batch_size': args["batch_size"],
        'num_workers': args["num_workers"],
        'shuffle': True,
        'collate_fn': collate_fn,
    }

    train_loader = DataLoader(train_dset, **loader_kwargs)

    loader_kwargs['shuffle'] = args["shuffle"]
    val_loader = DataLoader(val_dset, **loader_kwargs)
    return vocab, train_loader, val_loader

def build_model(args, vocab):
  kwargs = {
      'vocab': vocab,
      'image_size': args['image_size'],
      'embedding_dim': args['embedding_dim'],
      'gconv_dim': args['gconv_dim'],
      'gconv_hidden_dim': args['gconv_hidden_dim'],
      'gconv_num_layers': args['gconv_num_layers'],
      'mlp_normalization': args['mlp_normalization'],
      'refinement_dims': args['refinement_dims'],
      'normalization': args['normalization'],
      'activation': args['activation'],
      'mask_size': args['mask_size'],
      'layout_noise_dim': args['layout_noise_dim'],
    }
  model = GraphConvModel(**kwargs)



  model.to(torch.device("cuda:0"))
  return model, kwargs


def imagenet_deprocess_batch(v):
    pass


def check_model(args, t, loader, model):
  float_dtype = torch.cuda.FloatTensor
  long_dtype = torch.cuda.LongTensor
  num_samples = 0
  all_losses = defaultdict(list)
  total_iou = 0
  total_boxes = 0
  with torch.no_grad():
    for batch in loader:
      batch = [tensor.cuda() for tensor in batch]
      masks = None
      if len(batch) == 6:
        imgs, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 7:
        imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      predicates = triples[:, 1] 

      # Run the model as it has been run during training
      model_masks = masks
      model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=model_masks)
      boxes_pred, masks_pred = model_out

      skip_pixel_loss = False
      total_loss, losses =  calculate_model_losses(
                                args, skip_pixel_loss, model, imgs,
                                boxes, boxes_pred, masks, masks_pred,
                                predicates)

      total_iou += jaccard(boxes_pred.cpu().data.numpy().flatten(), boxes.cpu().data.numpy().flatten())
      total_boxes += boxes_pred.size(0)

      for loss_name, loss_val in losses.items():
        all_losses[loss_name].append(loss_val)
      num_samples += imgs.size(0)
      if num_samples >= args['num_val_samples']:
        break

    samples = {}
    samples['gt_img'] = imgs

    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes, masks_gt=masks)
    samples['gt_box_gt_mask'] = model_out[0]

    model_out = model(objs, triples, obj_to_img, boxes_gt=boxes)
    samples['gt_box_pred_mask'] = model_out[0]

    model_out = model(objs, triples, obj_to_img)
    samples['pred_box_pred_mask'] = model_out[0]

    for k, v in samples.items():
      samples[k] = imagenet_deprocess_batch(v)

    mean_losses = {k: np.mean(v) for k, v in all_losses.items()}
    avg_iou = total_iou / total_boxes

    masks_to_store = masks
    if masks_to_store is not None:
      masks_to_store = masks_to_store.data.cpu().clone()

    masks_pred_to_store = masks_pred
    if masks_pred_to_store is not None:
      masks_pred_to_store = masks_pred_to_store.data.cpu().clone()

  batch_data = {
    'objs': objs.detach().cpu().clone(),
    'boxes_gt': boxes.detach().cpu().clone(),
    'masks_gt': masks_to_store,
    'triples': triples.detach().cpu().clone(),
    'obj_to_img': obj_to_img.detach().cpu().clone(),
    'triple_to_img': triple_to_img.detach().cpu().clone(),
    'boxes_pred': boxes_pred.detach().cpu().clone(),
    'masks_pred': masks_pred_to_store
  }
  out = [mean_losses, samples, batch_data, avg_iou]

  return tuple(out)


def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args)
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor

    vocab, train_loader, val_loader = build_loaders(args)
    model, model_kwargs = build_model(args, vocab)
    model.type(float_dtype)

    ###########################################################################################
    print(device)
    model.to(device)
    print("model is cuda?")
    print(next(model.parameters()).is_cuda)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])

    t, epoch = 0, 0
    checkpoint = {
        'args': args,
        'vocab': vocab,
        'model_kwargs': model_kwargs,
        'losses_ts': [],
        'losses': defaultdict(list),
        'd_losses': defaultdict(list),
        'checkpoint_ts': [],
        'train_batch_data': [],
        'train_samples': [],
        'train_iou': [],
        'val_batch_data': [],
        'val_samples': [],
        'val_losses': defaultdict(list),
        'val_iou': [],
        'norm_d': [],
        'norm_g': [],
        'counters': {
            't': None,
            'epoch': None,
        },
        'model_state': None, 'model_best_state': None, 'optim_state': None,
        'd_obj_state': None, 'd_obj_best_state': None, 'd_obj_optim_state': None,
        'd_img_state': None, 'd_img_best_state': None, 'd_img_optim_state': None,
        'best_t': [],
    }

    while True:

        print('Starting epoch %d' % epoch)

        if t >= args['num_iterations']:
            break
        epoch += 1


        for batch in train_loader:

            if t == args['eval_mode_after']:
                print('switching to eval mode')
                model.eval()
                optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
            t += 1

            batch = [tensor.cuda() for tensor in batch]
            masks = None


            if len(batch) == 7:
                imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
                # print(batch)
            else:
                assert False

            predicates = triples[:, 1]

            # with timeit('forward', args['timing']): #args['timing']
            model_boxes = boxes
            model_masks = masks
            model_out = model(objs, triples, obj_to_img,
                              boxes_gt=model_boxes, masks_gt=model_masks)
            #imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
            boxes_pred, masks_pred = model_out
            #print(vector)

            skip_pixel_loss = (model_boxes is None)
            total_loss, losses = calculate_model_losses(
                args, skip_pixel_loss, model, imgs,
                boxes, boxes_pred, masks, masks_pred,
                predicates)



            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            #with timeit('backward', args.timing):
            total_loss.backward()
            optimizer.step()
            total_loss_d = None
            ac_loss_real = None
            ac_loss_fake = None
            d_losses = {}



            if t % args['print_every'] == 0:
                print('t = %d / %d' % (t, args['num_iterations']))
                for name, val in losses.items():
                    print(' G [%s]: %.4f' % (name, val))
                    checkpoint['losses'][name].append(val)
                checkpoint['losses_ts'].append(t)


            if t % args['checkpoint_every'] == 0:
                print('checking on train')
                train_results = check_model(args, t, train_loader, model)
                t_losses, t_samples, t_batch_data, t_avg_iou = train_results

                checkpoint['train_batch_data'].append(t_batch_data)
                checkpoint['train_samples'].append(t_samples)
                checkpoint['checkpoint_ts'].append(t)
                checkpoint['train_iou'].append(t_avg_iou)

                print('checking on val')
                val_results = check_model(args, t, val_loader, model)
                val_losses, val_samples, val_batch_data, val_avg_iou = val_results
                checkpoint['val_samples'].append(val_samples)
                checkpoint['val_batch_data'].append(val_batch_data)
                checkpoint['val_iou'].append(val_avg_iou)

                print('train iou: ', t_avg_iou)
                print('val iou: ', val_avg_iou)

                for k, v in val_losses.items():
                    checkpoint['val_losses'][k].append(v)
                checkpoint['model_state'] = model.state_dict()



                checkpoint['optim_state'] = optimizer.state_dict()
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint_path = os.path.join(args['output_dir'],
                                               '%s_with_model.pt' % args['checkpoint_name'])
                print('Saving checkpoint to ', checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

                # Save another checkpoint without any model or optim state
                checkpoint_path = os.path.join(args['output_dir'],
                                               '%s_no_model.pt' % args['checkpoint_name'])
                key_blacklist = ['model_state', 'optim_state', 'model_best_state',
                                 'd_obj_state', 'd_obj_optim_state', 'd_obj_best_state',
                                 'd_img_state', 'd_img_optim_state', 'd_img_best_state']
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)

    return


if __name__ == '__main__':

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

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
    }

    main(args)


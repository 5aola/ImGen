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

from prepareData.prepareData import coco_collate_fn, build_coco_dsets


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
  return model, kwargs

'''
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
      imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out

      skip_pixel_loss = False
      total_loss, losses =  calculate_model_losses(
                                args, skip_pixel_loss, model, imgs, imgs_pred,
                                boxes, boxes_pred, masks, masks_pred,
                                predicates, predicate_scores)

      total_iou += jaccard(boxes_pred, boxes)
      total_boxes += boxes_pred.size(0)

      for loss_name, loss_val in losses.items():
        all_losses[loss_name].append(loss_val)
      num_samples += imgs.size(0)
      if num_samples >= args.num_val_samples:
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
'''

def main(args):

    print(args)
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor

    vocab, train_loader, val_loader = build_loaders(args)
    model, model_kwargs = build_model(args, vocab)
    model.type(float_dtype)
    print(model)

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

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
            '''
            if t == args.eval_mode_after:
                print('switching to eval mode')
                model.eval()
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            t += 1
            '''
            batch = [tensor.cuda() for tensor in batch]
            masks = None


            if len(batch) == 7:
                imgs, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
            else:
                assert False

            #predicates = triples[:, 1]

            # with timeit('forward', args['timing']): #args['timing']
            model_boxes = boxes
            model_masks = masks
            model_out = model(objs, triples, obj_to_img,
                              boxes_gt=model_boxes, masks_gt=model_masks)
            #imgs_pred, boxes_pred, masks_pred, predicate_scores = model_out
            vector = model_out
            print(vector)

            '''    
            with timeit('loss', args.timing):
                # Skip the pixel loss if using GT boxes
                skip_pixel_loss = (model_boxes is None)
                total_loss, losses = calculate_model_losses(
                    args, skip_pixel_loss, model, imgs, imgs_pred,
                    boxes, boxes_pred, masks, masks_pred,
                    predicates, predicate_scores)



            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                print('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            with timeit('backward', args.timing):
                total_loss.backward()
            optimizer.step()
            total_loss_d = None
            ac_loss_real = None
            ac_loss_fake = None
            d_losses = {}



            if t % args.print_every == 0:
                print('t = %d / %d' % (t, args.num_iterations))
                for name, val in losses.items():
                    print(' G [%s]: %.4f' % (name, val))
                    checkpoint['losses'][name].append(val)
                checkpoint['losses_ts'].append(t)


            if t % args.checkpoint_every == 0:
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
                checkpoint_path = os.path.join(args.output_dir,
                                               '%s_with_model.pt' % args.checkpoint_name)
                print('Saving checkpoint to ', checkpoint_path)
                torch.save(checkpoint, checkpoint_path)

                # Save another checkpoint without any model or optim state
                checkpoint_path = os.path.join(args.output_dir,
                                               '%s_no_model.pt' % args.checkpoint_name)
                key_blacklist = ['model_state', 'optim_state', 'model_best_state',
                                 'd_obj_state', 'd_obj_optim_state', 'd_obj_best_state',
                                 'd_img_state', 'd_img_optim_state', 'd_img_best_state']
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                '''
    return


if __name__ == '__main__':

    print(torch.cuda.is_available())


    args = {
        'batch_size': 32,
        'num_workers': 4,
        'shuffle': True,
        'collate_fn': 10,

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
        'num_train_samples': 900,
        'num_val_samples': 100,
        'timing': '1',

    }

    main(args)

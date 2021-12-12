#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, json, os

from imageio import imwrite
import torch

import torchvision.transforms as T

from NagyHF.GraphConvModel import GraphConvModel
from NagyHF.prepareData.prepareData import build_coco_dsets
from NagyHF.utils import imagenet_deprocess_batch

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='sg2im-models/vg128.pt')
parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])


def main(args):
  vocab, train_dset, val_dset = build_coco_dsets()
  print(vocab)
  if not os.path.isfile(args['checkpoint']):
    print('ERROR: Checkpoint file "%s" not found' % args['checkpoint'])
    return

  if not os.path.isdir(args['output_dir']):
    print('Output directory "%s" does not exist; creating it' % args['output_dir'])
    os.makedirs(args['output_dir'])

  if args['device'] == 'cpu':
    device = torch.device('cpu')
  elif args['device'] == 'gpu':
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
      print('WARNING: CUDA not available; falling back to CPU')
      device = torch.device('cpu')

  # Load the model, with a bit of care in case there are no GPUs
  map_location = 'cpu' if device == torch.device('cpu') else None
  checkpoint = torch.load(args['checkpoint'], map_location=map_location)
  model = GraphConvModel(vocab)
  model.load_state_dict(checkpoint['model'])
  model.eval()
  model.to(device)

  # Load the scene graphs
  with open(args['scene_graphs_json'], 'r') as f:
    scene_graphs = json.load(f)

  # Run the model forward
  with torch.no_grad():
    boxes_pred, masks_pred, imgs = model.forward_json(scene_graphs)
  imgs = imagenet_deprocess_batch(imgs)

  # Save the generated images
  for i in range(imgs.shape[0]):

    img_np = imgs[i].numpy().transpose(1, 2, 0)
    #img_np = imgs[i].permute(1, 2, 0).cpu().detach().numpy()

    img_path = os.path.join(args['output_dir'], 'img%06d.png' % i)
    imwrite(img_path, img_np)



if __name__ == '__main__':
  args = {
    'checkpoint': os.path.join(os.getcwd(), 'model.pt'),
    'scene_graphs_json': os.path.join(os.getcwd(), 'scene_graphs/figure_6_sheep.json'),
    'output_dir': os.path.join(os.getcwd(), 'generatedImages'),
    'device': 'cpu',

  }

  #args = parser.parse_args()
  main(args)

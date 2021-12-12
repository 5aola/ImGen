#!/usr/bin/python


import argparse, json, os

from imageio import imwrite
import torch

import torchvision.transforms as T

from NagyHF.GraphConvModel import GraphConvModel
from NagyHF.prepareData.prepareData import build_coco_dsets
from NagyHF.utils import imagenet_deprocess_batch

# the function creates the objs, triples and obj_to_img from the given scene graphs
# scene graphs structure is visible below:
'''
  {
    "objects": ["lion", "giraffe", "sky"],
    "relationships": [
      [0, "next to", 1],
      [0, "beneath", 2],
      [2, "above", 1],
    ]
  }
'''


def encode_scene_graphs(vocab, scene_graphs):
    if isinstance(scene_graphs, dict):
        # We just got a single scene graph, so promote it to a list
        scene_graphs = [scene_graphs]

    objs, triples, obj_to_img = [], [], []
    obj_offset = 0
    for i, sg in enumerate(scene_graphs):
        # Insert dummy __image__ object and __in_image__ relationships
        sg['objects'].append('__image__')
        image_idx = len(sg['objects']) - 1
        for j in range(image_idx):
            sg['relationships'].append([j, '__in_image__', image_idx])

        for obj in sg['objects']:
            obj_idx = vocab['object_name_to_idx'].get(obj, None)
            if obj_idx is None:
                raise ValueError('Object "%s" not in vocab' % obj)
            objs.append(obj_idx)
            obj_to_img.append(i)
        for s, p, o in sg['relationships']:
            pred_idx = vocab['pred_name_to_idx'].get(p, None)
            if pred_idx is None:
                raise ValueError('Relationship "%s" not in vocab' % p)
            triples.append([s + obj_offset, pred_idx, o + obj_offset])
        obj_offset += len(sg['objects'])
    device = torch.device('cpu')
    objs = torch.tensor(objs, dtype=torch.int64, device=device)
    triples = torch.tensor(triples, dtype=torch.int64, device=device)
    obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
    return objs, triples, obj_to_img


def main(args):

    device = torch.device('cpu')
    checkpoint = torch.load(args['checkpoint'], map_location='cpu')

    # loading vocab
    vocab, train_dset = build_coco_dsets()

    #creating the model and loading its pretrained weights
    model = GraphConvModel(vocab)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    # Load the scene graphs
    with open(args['scene_graphs_json'], 'r') as f:
        scene_graphs = json.load(f)

    # Run the model forward
    with torch.no_grad():
        objs, triples, obj_to_img = encode_scene_graphs(vocab, scene_graphs)
        boxes_pred, masks_pred, imgs = model.forward(objs, triples, obj_to_img)
    imgs = imagenet_deprocess_batch(imgs)

    # Save the generated images
    for i in range(imgs.shape[0]):
        img_np = imgs[i].numpy().transpose(1, 2, 0)
        img_path = os.path.join(args['output_dir'], 'img%06d.png' % i)
        imwrite(img_path, img_np)


if __name__ == '__main__':
    # required arguments for the main function
    args = {
        'checkpoint': os.path.join(os.getcwd(), 'model.pt'),
        'scene_graphs_json': os.path.join(os.getcwd(), 'scene_graphs/inputSceneGraphs.json'),
        'output_dir': os.path.join(os.getcwd(), 'generatedImages'),
        'device': 'cpu',
    }
    main(args)

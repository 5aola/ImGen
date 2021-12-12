import json
import math
import os, random
from collections import defaultdict

import PIL
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import pycocotools.mask as mask_utils
from skimage.transform import resize as imresize

from NagyHF.utils import imagenet_preprocess, Resize


def build_coco_dsets():

  #we used our previously downloaded dataset
  dset_kwargs_train = {
    'image_dir': "C:/Users/kosty/Desktop/ImGen/ImGen/im_dataset/train2017",
    'instances_json': "C:/Users/kosty/Desktop/ImGen/ImGen/im_dataset/annotations_trainval2017/annotations/instances_train2017.json"
  }
  
  train_dset = CocoSceneGraphDataset(**dset_kwargs_train)
  num_objs = train_dset.total_objects()
  num_imgs = len(train_dset)
  print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
  print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

  vocab = json.loads(json.dumps(train_dset.vocab))

  return vocab, train_dset


class CocoSceneGraphDataset(Dataset):
  # for loading Coco dataset and converting them to scene graphs.
  def __init__(self, image_dir, instances_json):

    # parameters of the accepted images
    image_size = (64, 64)
    mask_size = 16
    min_object_size = 0.02
    min_objects_per_image = 2
    max_objects_per_image = 4
    instance_whitelist = ('ocean', 'sky', 'giraffe', 'lion', 'tree', 'person')
    super(Dataset, self).__init__()

    self.image_dir = image_dir
    self.mask_size = mask_size
    self.set_image_size(image_size)

    with open(instances_json, 'r') as f:
      instances_data = json.load(f)

    self.image_ids = []
    self.image_id_to_filename = {}
    self.image_id_to_size = {}
    for image_data in instances_data['images']:
      image_id = image_data['id']
      filename = image_data['file_name']
      width = image_data['width']
      height = image_data['height']
      self.image_ids.append(image_id)
      self.image_id_to_filename[image_id] = filename
      self.image_id_to_size[image_id] = (width, height)

    self.vocab = {
      'object_name_to_idx': {},
      'pred_name_to_idx': {},
    }

    object_idx_to_name = {}

    # filling the vocab with objects and relationships
    for category_data in instances_data['categories']:
      category_id = category_data['id']
      category_name = category_data['name']
      object_idx_to_name[category_id] = category_name
      self.vocab['object_name_to_idx'][category_name] = category_id


    category_whitelist = set(instance_whitelist)

    # Add object data from instances
    self.image_id_to_objects = defaultdict(list)
    for object_data in instances_data['annotations']:
      image_id = object_data['image_id']
      _, _, w, h = object_data['bbox']
      W, H = self.image_id_to_size[image_id]
      box_area = (w * h) / (W * H)
      box_ok = box_area > min_object_size
      object_name = object_idx_to_name[object_data['category_id']]
      category_ok = object_name in category_whitelist
      other_ok = object_name != 'other'
      if box_ok and category_ok and other_ok:
        self.image_id_to_objects[image_id].append(object_data)



    # COCO category labels start at 1, so use 0 for __image__
    self.vocab['object_name_to_idx']['__image__'] = 0

    # Build object_idx_to_name
    name_to_idx = self.vocab['object_name_to_idx']
    assert len(name_to_idx) == len(set(name_to_idx.values()))
    max_object_idx = max(name_to_idx.values())
    idx_to_name = ['NONE'] * (1 + max_object_idx)
    for name, idx in self.vocab['object_name_to_idx'].items():
      idx_to_name[idx] = name
    self.vocab['object_idx_to_name'] = idx_to_name

    # Prune images that have too few or too many objects
    new_image_ids = []
    total_objs = 0
    for image_id in self.image_ids:
      num_objs = len(self.image_id_to_objects[image_id])
      total_objs += num_objs
      if min_objects_per_image <= num_objs <= max_objects_per_image:
        new_image_ids.append(image_id)
    self.image_ids = new_image_ids

    self.vocab['pred_idx_to_name'] = [
      '__in_image__',
      'left of',
      'right of',
      'above',
      'below',
      'inside',
      'surrounding',
    ]

    self.vocab['pred_name_to_idx'] = {}
    for idx, name in enumerate(self.vocab['pred_idx_to_name']):
      self.vocab['pred_name_to_idx'][name] = idx

  #resizing, standardizing the images in the dataset
  def set_image_size(self, image_size):
    print('called set_image_size', image_size)
    transform = [Resize(image_size), T.ToTensor()]
    transform.append(imagenet_preprocess())
    self.transform = T.Compose(transform)
    self.image_size = image_size


  #calculate the object in the dataset
  def total_objects(self):
    total_objs = 0
    for i, image_id in enumerate(self.image_ids):
      num_objs = len(self.image_id_to_objects[image_id])
      total_objs += num_objs
    return total_objs


  def __len__(self):
    return len(self.image_ids)

  #getitem function returns the tuple of a certains image. The tuple contains the image (3, Height, Width), objects, boxes, masks and triples that contains the relatonships of the objects
  def __getitem__(self, index):

    image_id = self.image_ids[index]

    filename = self.image_id_to_filename[image_id]
    image_path = os.path.join(self.image_dir, filename)
    with open(image_path, 'rb') as f:
      with PIL.Image.open(f) as image:
        WW, HH = image.size
        image = self.transform(image.convert('RGB'))

    objs, boxes, masks = [], [], []
    for object_data in self.image_id_to_objects[image_id]:
      objs.append(object_data['category_id'])
      x, y, w, h = object_data['bbox']
      x0 = x / WW
      y0 = y / HH
      x1 = (x + w) / WW
      y1 = (y + h) / HH
      boxes.append(torch.FloatTensor([x0, y0, x1, y1]))

      # This will give a numpy array of shape (HH, WW)
      mask = seg_to_mask(object_data['segmentation'], WW, HH)

      # Crop the mask according to the bounding box, being careful to
      # ensure that we don't crop a zero-area region
      mx0, mx1 = int(round(x)), int(round(x + w))
      my0, my1 = int(round(y)), int(round(y + h))
      mx1 = max(mx0 + 1, mx1)
      my1 = max(my0 + 1, my1)
      mask = mask[my0:my1, mx0:mx1]
      mask = imresize(255.0 * mask, (self.mask_size, self.mask_size),
                      mode='constant')
      mask = torch.from_numpy((mask > 128).astype(np.int64))
      masks.append(mask)

    # Add __image__ object
    objs.append(self.vocab['object_name_to_idx']['__image__'])
    boxes.append(torch.FloatTensor([0, 0, 1, 1]))
    masks.append(torch.ones(self.mask_size, self.mask_size).long())

    objs = torch.LongTensor(objs)
    boxes = torch.stack(boxes, dim=0)
    masks = torch.stack(masks, dim=0)

    # Compute centers of all objects
    obj_centers = []
    _, MH, MW = masks.size()
    for i, obj_idx in enumerate(objs):
      x0, y0, x1, y1 = boxes[i]
      mask = (masks[i] == 1)
      xs = torch.linspace(x0, x1, MW).view(1, MW).expand(MH, MW)
      ys = torch.linspace(y0, y1, MH).view(MH, 1).expand(MH, MW)
      if mask.sum() == 0:
        mean_x = 0.5 * (x0 + x1)
        mean_y = 0.5 * (y0 + y1)
      else:
        mean_x = xs[mask].mean()
        mean_y = ys[mask].mean()
      obj_centers.append([mean_x, mean_y])
    obj_centers = torch.FloatTensor(obj_centers)

    # Add triples
    triples = []
    num_objs = objs.size(0)
    __image__ = self.vocab['object_name_to_idx']['__image__']
    real_objs = []
    if num_objs > 1:
      real_objs = (objs != __image__).nonzero().squeeze(1)
    for cur in real_objs:
      choices = [obj for obj in real_objs if obj != cur]
      if len(choices) == 0:
        break
      other = random.choice(choices)
      if random.random() > 0.5:
        s, o = cur, other
      else:
        s, o = other, cur


      # Check for inside / surrounding
      sx0, sy0, sx1, sy1 = boxes[s]
      ox0, oy0, ox1, oy1 = boxes[o]
      d = obj_centers[s] - obj_centers[o]
      theta = math.atan2(d[1], d[0])

      if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
        p = 'surrounding'
      elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
        p = 'inside'
      elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
        p = 'left of'
      elif -3 * math.pi / 4 <= theta < -math.pi / 4:
        p = 'above'
      elif -math.pi / 4 <= theta < math.pi / 4:
        p = 'right of'
      elif math.pi / 4 <= theta < 3 * math.pi / 4:
        p = 'below'
      p = self.vocab['pred_name_to_idx'][p]
      triples.append([s, p, o])

    # Add __in_image__ triples
    O = objs.size(0)
    in_image = self.vocab['pred_name_to_idx']['__in_image__']
    for i in range(O - 1):
      triples.append([i, in_image, O - 1])

    triples = torch.LongTensor(triples)

    return image, objs, boxes, masks, triples

# for decoding segmentation masks using the pycocotools API.
def seg_to_mask(seg, width=1.0, height=1.0):

  if type(seg) == list:
    rles = mask_utils.frPyObjects(seg, height, width)
    rle = mask_utils.merge(rles)
  elif type(seg['counts']) == list:
    rle = mask_utils.frPyObjects(seg, height, width)
  else:
    rle = seg
  return mask_utils.decode(rle)

# coco_collate_fn is a funtion that gets the batch and returns the images, objects, boxes, masks, triples, obj_to_img, triple_to_img
def coco_collate_fn(batch):
  all_imgs, all_objs, all_boxes, all_masks, all_triples = [], [], [], [], []
  all_obj_to_img, all_triple_to_img = [], []
  obj_offset = 0
  for i, (img, objs, boxes, masks, triples) in enumerate(batch):
    all_imgs.append(img[None])
    if objs.dim() == 0 or triples.dim() == 0:
      continue
    O, T = objs.size(0), triples.size(0)
    all_objs.append(objs)
    all_boxes.append(boxes)
    all_masks.append(masks)
    triples = triples.clone()
    triples[:, 0] += obj_offset
    triples[:, 2] += obj_offset
    all_triples.append(triples)

    all_obj_to_img.append(torch.LongTensor(O).fill_(i))
    all_triple_to_img.append(torch.LongTensor(T).fill_(i))
    obj_offset += O

  all_imgs = torch.cat(all_imgs)
  all_objs = torch.cat(all_objs)
  all_boxes = torch.cat(all_boxes)
  all_masks = torch.cat(all_masks)
  all_triples = torch.cat(all_triples)
  all_obj_to_img = torch.cat(all_obj_to_img)
  all_triple_to_img = torch.cat(all_triple_to_img)

  out = (all_imgs, all_objs, all_boxes, all_masks, all_triples,
         all_obj_to_img, all_triple_to_img)
  return out





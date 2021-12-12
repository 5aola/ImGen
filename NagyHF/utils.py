import PIL
import torch
import torchvision.transforms as T
import torch.nn.functional as F


# Calculated mean and std of the full image dataset
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]

# deprocessing the images in batches by inverting the standardization to the range of (0:255)
def imagenet_deprocess_batch(imgs, rescale=True):

  if isinstance(imgs, torch.autograd.Variable):
    imgs = imgs.data
  imgs = imgs.cpu().clone()
  deprocess_fn = imagenet_deprocess(rescale_image=rescale)
  imgs_de = []
  for i in range(imgs.size(0)):
    img_de = deprocess_fn(imgs[i])[None]
    img_de = img_de.mul(255).clamp(0, 255).byte()
    imgs_de.append(img_de)
  imgs_de = torch.cat(imgs_de, dim=0)
  return imgs_de



def rescale(x):
  lo, hi = x.min(), x.max()
  return x.sub(lo).div(hi - lo)


# deprocessing one image by inverting the standardization to the range of (0:255)
def imagenet_deprocess(rescale_image=True):
  transforms = [
    T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
    T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
  ]
  if rescale_image:
    transforms.append(rescale)
  return T.Compose(transforms)


# tranforming the bounding boxes to a layout
def boxes_to_layout(vecs, boxes, obj_to_img, H, W=None, pooling='sum'):

    O, D = vecs.size()
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)
    img_in = vecs.view(O, D, 1, 1).expand(O, D, 8, 8)
    sampled = F.grid_sample(img_in, grid)  # (O, D, H, W)

    out = _pool_samples(sampled, obj_to_img)

    return out

# tranforming the segmentation masks to a layout
def masks_to_layout(vecs, boxes, masks, obj_to_img, H, W=None, pooling='sum'):
    O, D = vecs.size()
    M = masks.size(1)
    assert masks.size() == (O, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    img_in = vecs.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)
    sampled = F.grid_sample(img_in, grid)

    out = _pool_samples(sampled, obj_to_img)
    return out

def _boxes_to_grid(boxes, H, W):
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    x1, y1 = boxes[:, 2], boxes[:, 3]
    ww = x1 - x0
    hh = y1 - y0

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid

def _pool_samples(samples, obj_to_img):
    dtype, device = samples.dtype, samples.device
    O, D, H, W = samples.size()
    N = obj_to_img.data.max().item() + 1

    # Use scatter_add to sum the sampled outputs for each image
    out = torch.zeros(N, D, H, W, dtype=dtype, device=device)
    idx = obj_to_img.view(O, 1, 1, 1).expand(O, D, H, W)
    out = out.scatter_add(0, idx, samples)

    # Divide each output mask by the number of objects; use scatter_add again
    # to count the number of objects per image.
    ones = torch.ones(O, dtype=dtype, device=device)
    obj_counts = torch.zeros(N, dtype=dtype, device=device)
    obj_counts = obj_counts.scatter_add(0, obj_to_img, ones)
    #print(obj_counts)
    obj_counts = obj_counts.clamp(min=1)
    out = out / obj_counts.view(N, 1, 1, 1)

    return out

# Normalize images for model training
def imagenet_preprocess():
  IMAGENET_MEAN = [0.485, 0.456, 0.406]
  IMAGENET_STD = [0.229, 0.224, 0.225]
  return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# using bilinear interpolation for resizing the images to (64,64)
class Resize(object):
  def __init__(self, size, interp=PIL.Image.BILINEAR):
    if isinstance(size, tuple):
      H, W = size
      self.size = (W, H)
    else:
      self.size = (size, size)
    self.interp = interp

  def __call__(self, img):
    return img.resize(self.size, self.interp)


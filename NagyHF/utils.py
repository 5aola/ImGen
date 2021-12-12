import PIL
import torch
import torchvision.transforms as T
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_deprocess_batch(imgs, rescale=True):
  """
  Input:
  - imgs: FloatTensor of shape (N, C, H, W) giving preprocessed images

  Output:
  - imgs_de: ByteTensor of shape (N, C, H, W) giving deprocessed images
    in the range [0, 255]
  """
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


def imagenet_deprocess(rescale_image=True):
  transforms = [
    T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
    T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
  ]
  if rescale_image:
    transforms.append(rescale)
  return T.Compose(transforms)



def boxes_to_layout(vecs, boxes, obj_to_img, H, W=None, pooling='sum'):
    """
  Inputs:
  - vecs: Tensor of shape (O, D) giving vectors
  - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
    [x0, y0, x1, y1] in the [0, 1] coordinate space
  - obj_to_img: LongTensor of shape (O,) mapping each element of vecs to
    an image, where each element is in the range [0, N). If obj_to_img[i] = j
    then vecs[i] belongs to image j.
  - H, W: Size of the output

  Returns:
  - out: Tensor of shape (N, D, H, W)
  """
    O, D = vecs.size()
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    # If we don't add extra spatial dimensions here then out-of-bounds
    # elements won't be automatically set to 0
    img_in = vecs.view(O, D, 1, 1).expand(O, D, 8, 8)
    sampled = F.grid_sample(img_in, grid)  # (O, D, H, W)

    # Explicitly masking makes everything quite a bit slower.
    # If we rely on implicit masking the interpolated boxes end up
    # blurred around the edges, but it should be fine.
    # mask = ((X < 0) + (X > 1) + (Y < 0) + (Y > 1)).clamp(max=1)
    # sampled[mask[:, None]] = 0

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)

    return out


def masks_to_layout(vecs, boxes, masks, obj_to_img, H, W=None, pooling='sum'):
    """
  Inputs:
  - vecs: Tensor of shape (O, D) giving vectors
  - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
    [x0, y0, x1, y1] in the [0, 1] coordinate space
  - masks: Tensor of shape (O, M, M) giving binary masks for each object
  - obj_to_img: LongTensor of shape (O,) mapping objects to images
  - H, W: Size of the output image.

  Returns:
  - out: Tensor of shape (N, D, H, W)
  """
    O, D = vecs.size()
    M = masks.size(1)
    assert masks.size() == (O, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes, H, W)

    img_in = vecs.view(O, D, 1, 1) * masks.float().view(O, 1, M, M)
    sampled = F.grid_sample(img_in, grid)

    out = _pool_samples(sampled, obj_to_img, pooling=pooling)
    return out


def _boxes_to_grid(boxes, H, W):
    """
  Input:
  - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
    format in the [0, 1] coordinate space
  - H, W: Scalars giving size of output

  Returns:
  - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
  """
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


def _pool_samples(samples, obj_to_img, pooling='sum'):
    """
  Input:
  - samples: FloatTensor of shape (O, D, H, W)
  - obj_to_img: LongTensor of shape (O,) with each element in the range
    [0, N) mapping elements of samples to output images

  Output:
  - pooled: FloatTensor of shape (N, D, H, W)
  """
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


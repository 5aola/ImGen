import math
import torch
import torch.nn as nn

from NagyHF.crn import RefinementNetwork
from NagyHF.utils import masks_to_layout, boxes_to_layout
from graph import GraphTripleConvNet, build_mlp
from matplotlib import pyplot as plt


class GraphConvModel(nn.Module):

    def __init__(self, vocab):
        super(GraphConvModel, self).__init__()

        self.vocab = vocab
        self.image_size = (64, 64)
        self.embedding_dim = 128
        self.gconv_dim = 128
        self.gconv_hidden_dim = 512
        self.gconv_pooling = 'avg'
        self.gconv_num_layers = 5
        self.refinement_dims = (1024, 512, 256, 128, 64)
        self.normalization = 'batch'
        self.activation = 'leakyrelu-0.2'  # relu
        self.mask_size = 16
        self.mlp_normalization = 'batch'
        self.layout_noise_dim = 32

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])


        self.obj_embeddings = nn.Embedding(num_objs + 1, self.embedding_dim)
        self.pred_embeddings = nn.Embedding(num_preds, self.embedding_dim)

        gconv_kwargs = {
            'input_dim': self.gconv_dim,
            'hidden_dim': self.gconv_hidden_dim,
            'pooling': self.gconv_pooling,
            'num_layers': self.gconv_num_layers - 1,
            'mlp_normalization': self.mlp_normalization,
        }
        self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        box_net_dim = 4
        box_net_layers = [self.gconv_dim, self.gconv_hidden_dim, box_net_dim]
        self.box_net = build_mlp(box_net_layers, batch_norm=self.mlp_normalization)

        self.mask_net = self._build_mask_net(num_objs, self.gconv_dim, self.mask_size)

        refinement_kwargs = {
            'dims': (self.gconv_dim + self.layout_noise_dim,) + self.refinement_dims,
            'normalization': self.normalization,
            'activation': self.activation,
        }

        self.refinement_net = RefinementNetwork(**refinement_kwargs)

    def _build_mask_net(self, num_objs, dim, mask_size):
        output_dim = 1
        layers, cur_size = [], 1
        while cur_size < mask_size:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.BatchNorm2d(dim))
            layers.append(nn.Conv2d(dim, dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            cur_size *= 2
        if cur_size != mask_size:
            raise ValueError('Mask size must be a power of 2')
        layers.append(nn.Conv2d(dim, output_dim, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, objs, triples, obj_to_img=None,
                boxes_gt=None, masks_gt=None):
        """
        Required Inputs:
        - objs: LongTensor of shape (O,) giving categories for all objects
        - triples: LongTensor of shape (T, 3) where triples[t] = [s, p, o]
          means that there is a triple (objs[s], p, objs[o])

        Optional Inputs:
        - obj_to_img: LongTensor of shape (O,) where obj_to_img[o] = i
          means that objects[o] is an object in image i. If not given then
          all objects are assumed to belong to the same image.
        - boxes_gt: FloatTensor of shape (O, 4) giving boxes to use for computing
          the spatial layout; if not given then use predicted boxes.
        """

        O, T = objs.size(0), triples.size(0)
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        obj_vecs_orig = obj_vecs
        pred_vecs = self.pred_embeddings(p)

        obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        boxes_pred = self.box_net(obj_vecs)

        masks_pred = None
        if self.mask_net is not None:
            mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
            masks_pred = mask_scores.squeeze(1).sigmoid()


        H, W = self.image_size

        layout_boxes = boxes_pred if boxes_gt is None else boxes_gt

        if masks_pred is None:
            layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
        else:
            layout_masks = masks_pred if masks_gt is None else masks_gt
            layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
                                     obj_to_img, H, W)

        layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks, obj_to_img, H, W)


        #plt.figure()
        #plt.imshow(layout[0][0].permute(0, 1).cpu().detach().numpy(), interpolation='nearest')
        #plt.show()



        N, C, H, W = layout.size()
        noise_shape = (N, self.layout_noise_dim, H, W)
        layout_noise = torch.randn(noise_shape, dtype=layout.dtype,  device=layout.device)
        layout = torch.cat([layout, layout_noise], dim=1)
        img = self.refinement_net(layout)

        return boxes_pred, masks_pred, img

    def encode_scene_graphs(self, scene_graphs):
        """
        Encode one or more scene graphs using this model's vocabulary. Inputs to
        this method are scene graphs represented as dictionaries like the following:

        {
          "objects": ["cat", "dog", "sky"],
          "relationships": [
            [0, "next to", 1],
            [0, "beneath", 2],
            [2, "above", 1],
          ]
        }

        This scene graph has three relationshps: cat next to dog, cat beneath sky,
        and sky above dog.

        Inputs:
        - scene_graphs: A dictionary giving a single scene graph, or a list of
          dictionaries giving a sequence of scene graphs.

        Returns a tuple of LongTensors (objs, triples, obj_to_img) that have the
        same semantics as self.forward. The returned LongTensors will be on the
        same device as the model parameters.
        """
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
                obj_idx = self.vocab['object_name_to_idx'].get(obj, None)
                if obj_idx is None:
                    raise ValueError('Object "%s" not in vocab' % obj)
                objs.append(obj_idx)
                obj_to_img.append(i)
            for s, p, o in sg['relationships']:
                pred_idx = self.vocab['pred_name_to_idx'].get(p, None)
                if pred_idx is None:
                    raise ValueError('Relationship "%s" not in vocab' % p)
                triples.append([s + obj_offset, pred_idx, o + obj_offset])
            obj_offset += len(sg['objects'])
        device = next(self.parameters()).device
        objs = torch.tensor(objs, dtype=torch.int64, device=device)
        triples = torch.tensor(triples, dtype=torch.int64, device=device)
        obj_to_img = torch.tensor(obj_to_img, dtype=torch.int64, device=device)
        return objs, triples, obj_to_img

    def forward_json(self, scene_graphs):
        """ Convenience method that combines encode_scene_graphs and forward. """
        objs, triples, obj_to_img = self.encode_scene_graphs(scene_graphs)
        return self.forward(objs, triples, obj_to_img)

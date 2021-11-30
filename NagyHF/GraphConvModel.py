import math
import torch
import torch.nn as nn

from graph import GraphTripleConvNet


class GraphConvModel(nn.Module):

    def __init__(self, vocab, image_size=(64, 64), embedding_dim=64,
                 gconv_dim=128, gconv_hidden_dim=512,
                 gconv_pooling='avg', gconv_num_layers=5,
                 refinement_dims=(1024, 512, 256, 128, 64),
                 normalization='batch', activation='leakyrelu-0.2',
                 mask_size=None, mlp_normalization='none', layout_noise_dim=0,
                 **kwargs):
        super(GraphConvModel, self).__init__()

        # We used to have some additional arguments:
        # vec_noise_dim, gconv_mode, box_anchor, decouple_obj_predictions
        if len(kwargs) > 0:
            print('WARNING: Model got unexpected kwargs ', kwargs)

        self.vocab = vocab

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])
        self.obj_embeddings = nn.Embedding(num_objs + 1, embedding_dim)
        self.pred_embeddings = nn.Embedding(num_preds, embedding_dim)

        gconv_kwargs = {
            'input_dim': gconv_dim,
            'hidden_dim': gconv_hidden_dim,
            'pooling': gconv_pooling,
            'num_layers': gconv_num_layers - 1,
            'mlp_normalization': mlp_normalization,
        }
        self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

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

        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)
        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)
        obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        return obj_vecs, pred_vecs

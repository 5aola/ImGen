import torch
import torch.nn as nn

from NagyHF.box_net import BoxNet
from NagyHF.crn import RefinementNetwork
from NagyHF.seg_mask_net import MaskNet
from NagyHF.utils import masks_to_layout, boxes_to_layout
from graph import GraphTripleConvNet
from matplotlib import pyplot as plt

# GraphConvModel represents the whole model
class GraphConvModel(nn.Module):

    def __init__(self, vocab):
        super(GraphConvModel, self).__init__()


        # Adding default parameters for the network

        self.vocab = vocab
        self.image_size = (64, 64)
        self.embedding_dim = 128
        self.gconv_dim = 128
        self.gconv_hidden_dim = 512
        self.gconv_pooling = 'avg'
        self.gconv_num_layers = 5
        self.refinement_dims = (1024, 512, 256, 128, 64)
        self.activation = 'leakyrelu-0.2'
        self.mask_size = 16
        self.mlp_normalization = 'batch'
        self.layout_noise_dim = 32

        num_objs = len(vocab['object_idx_to_name'])
        num_preds = len(vocab['pred_idx_to_name'])

        # Embedding the objects and predictions to embedding_dim dimension
        self.obj_embeddings = nn.Embedding(num_objs + 1, self.embedding_dim)
        self.pred_embeddings = nn.Embedding(num_preds, self.embedding_dim)

        # the parameters of the graph conv network
        gconv_kwargs = {
            'input_dim': self.gconv_dim,
            'hidden_dim': self.gconv_hidden_dim,
            'pooling': self.gconv_pooling,
            'num_layers': self.gconv_num_layers - 1,
            'mlp_normalization': self.mlp_normalization,
        }

        # building the graph convolutional network
        self.gconv_net = GraphTripleConvNet(**gconv_kwargs)

        # building an MLP network for the box predictions
        box_net_dim = 4
        box_net_layers = [self.gconv_dim, self.gconv_hidden_dim, box_net_dim]
        self.box_net = BoxNet(box_net_layers, batch_norm=self.mlp_normalization)

        # building a CNN network for the mask predictions
        self.mask_net = MaskNet(self.gconv_dim, self.mask_size) #self._build_mask_net(num_objs, self.gconv_dim, self.mask_size)

        # the parameters of the cascaded refinement network
        refinement_kwargs = {
            'dims': (self.gconv_dim + self.layout_noise_dim,) + self.refinement_dims,
            'activation': self.activation,
        }

        # building a CRN network for the image synthesis
        self.refinement_net = RefinementNetwork(**refinement_kwargs)



    # The main forward method
    def forward(self, objs, triples, obj_to_img=None, boxes_gt=None, masks_gt=None):

        O, T = objs.size(0), triples.size(0)

        # Splitting the triples
        s, p, o = triples.chunk(3, dim=1)  # All have shape (T, 1)
        s, p, o = [x.squeeze(1) for x in [s, p, o]]  # Now have shape (T,)
        edges = torch.stack([s, o], dim=1)  # Shape is (T, 2)

        if obj_to_img is None:
            obj_to_img = torch.zeros(O, dtype=objs.dtype, device=objs.device)

        obj_vecs = self.obj_embeddings(objs)
        pred_vecs = self.pred_embeddings(p)

        obj_vecs, pred_vecs = self.gconv_net(obj_vecs, pred_vecs, edges)

        boxes_pred = self.box_net(obj_vecs)

        masks_pred = None
        if self.mask_net is not None:
            mask_scores = self.mask_net(obj_vecs.view(O, -1, 1, 1))
            masks_pred = mask_scores.squeeze(1).sigmoid()


        H, W = self.image_size

        # For more efficent image synthesis learning, we are using the original layout boxes, to train the refinement_net,
        # if the masks predictions are not accurate
        layout_boxes = boxes_pred if boxes_gt is None else boxes_gt

        if masks_pred is None:
            layout = boxes_to_layout(obj_vecs, layout_boxes, obj_to_img, H, W)
        else:
            layout_masks = masks_pred if masks_gt is None else masks_gt
            layout = masks_to_layout(obj_vecs, layout_boxes, layout_masks,
                                     obj_to_img, H, W)

        # for inspecting the maskr uncomment the below code snippet
        #plt.figure()
        #plt.imshow(layout[0][0].permute(0, 1).cpu().detach().numpy(), interpolation='nearest')
        #plt.show()

        N, C, H, W = layout.size()
        noise_shape = (N, self.layout_noise_dim, H, W)
        layout_noise = torch.randn(noise_shape, dtype=layout.dtype,  device=layout.device)
        layout = torch.cat([layout, layout_noise], dim=1)
        img = self.refinement_net(layout)

        return boxes_pred, masks_pred, img



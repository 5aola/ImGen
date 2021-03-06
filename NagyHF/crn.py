import torch
import torch.nn as nn
import torch.nn.functional as F



# creates activation layer based on the input
def get_activation(name):
  kwargs = {}
  if name.lower().startswith('leakyrelu'):
    if '-' in name:
      slope = float(name.split('-')[1])
      kwargs = {'negative_slope': slope}
  name = 'leakyrelu'
  activations = {
    'relu': nn.ReLU,
    'leakyrelu': nn.LeakyReLU,
  }
  if name.lower() not in activations:
    raise ValueError('Invalid activation "%s"' % name)
  return activations[name.lower()](**kwargs)


# A sequence of RefinementModule layers
class RefinementNetwork(nn.Module):
  def __init__(self, dims, activation='leakyrelu'):
    super(RefinementNetwork, self).__init__()
    layout_dim = dims[0]
    self.refinement_modules = nn.ModuleList()
    for i in range(1, len(dims)):
      input_dim = 1 if i == 1 else dims[i - 1]
      output_dim = dims[i]
      mod = RefinementModule(layout_dim, input_dim, output_dim, activation=activation)
      self.refinement_modules.append(mod)
    output_conv_layers = [
        nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
        get_activation(activation),
        nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
        get_activation(activation),
        nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0)
    ]
    nn.init.kaiming_normal_(output_conv_layers[0].weight)
    nn.init.kaiming_normal_(output_conv_layers[2].weight)
    self.output_conv = nn.Sequential(*output_conv_layers)

  def forward(self, layout):
    """
    Output will have same size as layout
    """
    # H, W = self.output_size
    N, _, H, W = layout.size()
    self.layout = layout

    # Figure out size of input
    input_H, input_W = H, W
    for _ in range(len(self.refinement_modules)):
      input_H //= 2
      input_W //= 2

    assert input_H != 0
    assert input_W != 0

    feats = torch.zeros(N, 1, input_H, input_W).to(layout)
    for mod in self.refinement_modules:
      feats = F.upsample(feats, scale_factor=2, mode='nearest')
      feats = mod(layout, feats)

    out = self.output_conv(feats)
    return out



# A single layer of Refinement Module, that contains BatchNorm2d and Convolutional layers
class RefinementModule(nn.Module):
    def __init__(self, layout_dim, input_dim, output_dim, activation='leakyrelu'):
        super(RefinementModule, self).__init__()

        layers = []
        layers.append(nn.Conv2d(layout_dim + input_dim, output_dim,
                                kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(output_dim))
        layers.append(get_activation(activation))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(output_dim))
        layers.append(get_activation(activation))
        layers = [layer for layer in layers if layer is not None]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
        self.net = nn.Sequential(*layers)

    def forward(self, layout, feats):
        _, _, HH, WW = layout.size()
        _, _, H, W = feats.size()
        assert HH >= H
        if HH > H:
            factor = round(HH // H)
            assert HH % factor == 0
            assert WW % factor == 0 and WW // factor == W
            layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)
        net_input = torch.cat([layout, feats], dim=1)
        out = self.net(net_input)
        return out
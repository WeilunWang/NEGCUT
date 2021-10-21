import torch
import torch.nn as nn
from models.utils import init_net


def define_N(nce_layers, netN, init_type='normal', init_gain=0.02, gpu_ids=[], opt=None):
    if netN == 'neg_param':
        net = Negative_Placeholder(nce_layers, opt.num_patches, opt.netF_nc, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    elif netN == 'neg_gen':
        net = Negative_Generator(use_conv=True, num_patches=opt.num_patches, nc=opt.netF_nc, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    elif netN == 'neg_gen_al' or netN == 'neg_gen_momentum':
        net = Negative_Generator(use_conv=False, num_patches=opt.num_patches, nc=opt.netF_nc, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netN)
    return init_net(net, init_type='xavier', init_gain=1.0, gpu_ids=gpu_ids)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class Negative_Placeholder(nn.Module):
    def __init__(self, nce_layers, num_patches=256, nc=256, init_type='normal', init_gain=0.02, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(Negative_Placeholder, self).__init__()
        self.l2norm = Normalize(2)
        self.nce_layers = nce_layers
        self.num_patches = num_patches
        self.nc = nc  # hard-coded
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.neg_sample = nn.Parameter(torch.FloatTensor(len(nce_layers), 1, num_patches, nc), requires_grad=True)
        nn.init.xavier_normal_(self.neg_sample, gain=1.0)

    def forward(self, nce_layers, num_images):
        return_feats = []
        for layer_id, layer in enumerate(nce_layers):
            neg_sample = self.neg_sample[layer_id].repeat(num_images, 1, 1)
            neg_sample = neg_sample.view(-1, self.nc)
            neg_sample = self.l2norm(neg_sample)
            return_feats.append(neg_sample)
        return return_feats


class Negative_Generator(nn.Module):
    def __init__(self, use_conv=False, num_patches=256, nc=256, z_dim=64, init_type='normal', init_gain=0.02, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(Negative_Generator, self).__init__()
        self.l2norm = Normalize(2)
        self.num_patches = num_patches
        self.nc = nc
        self.z_dim = z_dim
        self.use_conv = use_conv
        self.layer_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_layers(self, feats):
        for feat_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            if self.use_conv:
                conv = nn.Sequential(*[nn.Conv2d(input_nc, self.nc, 1, 1), nn.ReLU(), nn.Conv2d(self.nc, self.nc, 1, 1)])
                if len(self.gpu_ids) > 0:
                    conv.cuda()
                setattr(self, 'conv_%d' % feat_id, conv)
            mlp = nn.Sequential(*[nn.Linear(self.nc + self.z_dim, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            if len(self.gpu_ids) > 0:
                mlp.cuda()
            setattr(self, 'mlp_%d' % feat_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.layer_init = True

    def forward(self, feats, num_patches):
        self.return_feats = []
        self.return_noise = []
        if not self.layer_init:
            self.create_layers(feats)
        for feat_id, feat in enumerate(feats):
            noise = torch.randn([feat.size(0), self.num_patches, self.z_dim])
            if torch.cuda.is_available():
                noise = noise.cuda()
            if self.use_conv:
                conv = getattr(self, 'conv_%d' % feat_id)
                feat = conv(feat)
            feat = feat.permute(0, 2, 3, 1).mean(dim=(1, 2))
            feat = feat.unsqueeze(dim=1).repeat(1, num_patches, 1)
            inp = torch.cat([feat, noise], dim=2).flatten(0, 1)
            mlp = getattr(self, 'mlp_%d' % feat_id)
            neg_sample = mlp(inp)
            neg_sample = self.l2norm(neg_sample)
            self.return_feats.append(neg_sample)
            self.return_noise.append(noise)
        return self.return_feats

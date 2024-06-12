import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder
from models.spherical import get_ide
class SirenLayer(nn.Linear):
    def __init__(self, input_dim, out_dim, *args, is_first=False, **kwargs):
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = 30
        self.c = 6
        super().__init__(input_dim, out_dim, *args, **kwargs)
        self.activation = Sine(self.w0)

    # override
    def reset_parameters(self) -> None:
        # NOTE: in offical SIREN, first run linear's original initialization, then run custom SIREN init.
        #       hence the bias is initalized in super()'s reset_parameters()
        super().reset_parameters()
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.weight.uniform_(-w_std, w_std)

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out
    
class DenseLayer(nn.Linear):
    def __init__(self, input_dim: int, out_dim: int, *args, activation=None, **kwargs):
        super().__init__(input_dim, out_dim, *args, **kwargs)
        if activation is None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        out = super().forward(x)
        out = self.activation(out)
        return out

# This implementation is borrowed from nerf-pytorch: https://github.com/yenchenlin/nerf-pytorch
class NeRF(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 d_in_view=3,
                 multires=0,
                 multires_view=0,
                 output_ch=4,
                 skips=[4],
                 use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        ### Implementation according to the official code release
        ### (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)


class RoughNet(nn.Module):
    def __init__(self,
        D=4,
        W=512,
        skips=[],
        W_geo_feat=256,
        embed_multires=10,
        weight_norm=True,
        use_siren=False,
        final_act='sigmoid'):
        super().__init__()
        
        input_ch_pts = 3
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
        self.skips = skips
        self.D = D
        self.W = W
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        in_dim_0 = input_ch_pts + W_geo_feat
        
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D + 1):
            # decicde out_dim
            if l == D:
                out_dim = 1
            else:
                out_dim = W
            
            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + W
            else:
                in_dim = W
            
            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l==0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                if final_act == 'sigmoid':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
                elif final_act == 'swish':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.SiLU())
                elif final_act == 'softplus':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus())
            
            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        geometry_feature: torch.Tensor):
        # calculate radiance field
        x = self.embed_fn(x)
        radiance_input = torch.cat([x, geometry_feature], dim=-1)
        
        h = radiance_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h
    
class IntEnvMapNet(nn.Module):
    def __init__(self,
        embedding_fn=get_ide,
        D=4,
        W=512,
        skips=[],
        embed_multires_view=2,
        weight_norm=True,
        use_siren=False,
        final_act='sigmoid'):
        super().__init__()

        # input_ch_pts = 3
        input_ch_views = 3
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
        self.skips = skips
        self.D = D
        self.W = W

        if embedding_fn == None:
            self.embed_fn_view, input_ch_views = get_embedder(embed_multires_view)
            in_dim_0 = input_ch_views + 1
            self.sh_embed = False
        else:
            self.embed_fn_view = embedding_fn 
            self.L = embed_multires_view
            in_dim_0 = 2**(self.L+2) + self.L - 1
            self.sh_embed = True

        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D + 1):
            # decicde out_dim
            if l == D:
                out_dim = 3
            else:
                out_dim = W

            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + W
            else:
                in_dim = W

            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l==0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                if final_act == 'sigmoid':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
                elif final_act == 'swish':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.SiLU())
                elif final_act == 'relu':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU())
                elif final_act == 'identity':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Identity())
                elif final_act == 'softplus':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus())

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)

    def forward(
        self, 
        view_dirs: torch.Tensor,
        roughness: torch.Tensor):

        if self.sh_embed:
            radiance_input = self.embed_fn_view(dirs=view_dirs, roughness=roughness[...,0], L=self.L)
        else:
            radiance_input = torch.cat([self.embed_fn_view(view_dirs), roughness], dim=-1)

        h = radiance_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h

class ImplicitSurface(nn.Module):
    def __init__(self,
                 W=256,
                 D=8,
                 skips=[4],
                 W_geo_feat=256,
                 input_ch=3,
                 radius_init=0.5,
                 obj_bounding_size=2.0,
                 geometric_init=True,
                 embed_multires=6,
                 weight_norm=True,
                 use_siren=False,
                 ):
        """
        W_geo_feat: to set whether to use nerf-like geometry feature or IDR-like geometry feature.
            set to -1: nerf-like, the output feature is the second to last level's feature of the geometry network.
            set to >0: IDR-like ,the output feature is the last part of the geometry network's output.
        """
        super().__init__()
        self.radius_init = radius_init
        self.register_buffer('obj_bounding_size', torch.tensor([obj_bounding_size]).float())
        self.geometric_init = geometric_init
        self.D = D
        self.W = W
        self.W_geo_feat = W_geo_feat
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
            self.register_buffer('is_pretrained', torch.tensor([False], dtype=torch.bool))
        self.skips = skips
        self.use_siren = use_siren
        self.embed_fn, input_ch = get_embedder(embed_multires)

        surface_fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D+1):
            # decide out_dim
            if l == D:
                if W_geo_feat > 0:
                    out_dim = 1 + W_geo_feat
                else:
                    out_dim = 1
            elif (l+1) in self.skips:
                out_dim = W - input_ch  # recude output dim before the skips layers, as in IDR / NeuS
            else:
                out_dim = W
                
            # decide in_dim
            if l == 0:
                in_dim = input_ch
            else:
                in_dim = W
            
            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first = (l==0))
                else:
                    # NOTE: beta=100 is important! Otherwise, the initial output would all be > 10, and there is not initial sphere.
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus(beta=100))
            else:
                layer = nn.Linear(in_dim, out_dim)

            # if true preform preform geometric initialization
            if geometric_init and not use_siren:
                #--------------
                # sphere init, as in SAL / IDR.
                #--------------
                if l == D:
                    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    nn.init.constant_(layer.bias, -radius_init) 
                elif embed_multires > 0 and l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.constant_(layer.weight[:, 3:], 0.0)   # let the initial weights for octaves to be 0.
                    torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif embed_multires > 0 and l in self.skips:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(layer.weight[:, -(input_ch - 3):], 0.0) # NOTE: this contrains the concat order to be  [h, x_embed]
                else:
                    nn.init.constant_(layer.bias, 0.0)
                    nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            surface_fc_layers.append(layer)

        self.surface_fc_layers = nn.ModuleList(surface_fc_layers)

    def forward(self, x: torch.Tensor):
        x = self.embed_fn(x)
        
        h = x
        for i in range(self.D):
            if i in self.skips:
                # NOTE: concat order can not change! there are special operations taken in intialization.
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = self.surface_fc_layers[i](h)
        
        out = self.surface_fc_layers[-1](h)
        

        return out
    
    def sdf(self, x):
        return self.forward(x)[:, :1]
    
    def forward_with_nablas(self,  x: torch.Tensor, has_grad_bypass: bool = None):
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass
        # force enabling grad for normal calculation
        with torch.enable_grad():
            x = x.requires_grad_(True)
            implicit_surface_val, h = self.forward(x, return_h=True)
            nabla = torch.autograd.grad(
                implicit_surface_val,
                x,
                torch.ones_like(implicit_surface_val, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True)[0]
        if not has_grad:
            implicit_surface_val = implicit_surface_val.detach()
            nabla = nabla.detach()
            h = h.detach()
        return implicit_surface_val, nabla, h
    
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    
class DiffuseNet(nn.Module):
    def __init__(self,
        D=4,
        W=512,
        skips=[],
        W_geo_feat=256,
        embed_multires=10,
        weight_norm=True,
        use_siren=False,
        final_act='sigmoid'):
        super().__init__()
        
        input_ch_pts = 3
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
        self.skips = skips
        self.D = D
        self.W = W
        self.embed_fn, input_ch_pts = get_embedder(embed_multires)
        in_dim_0 = input_ch_pts + W_geo_feat
        
        fc_layers = []
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D + 1):
            # decicde out_dim
            if l == D:
                out_dim = 3
            else:
                out_dim = W
            
            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + W
            else:
                in_dim = W
            
            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l==0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                if final_act == 'sigmoid':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
                elif final_act == 'swish':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.SiLU())
                elif final_act == 'softplus':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus())
            
            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)
    
    def forward(
        self, 
        x: torch.Tensor, 
        geometry_feature: torch.Tensor):
        # calculate radiance field
        x = self.embed_fn(x)
        radiance_input = torch.cat([x, geometry_feature], dim=-1)
        
        h = radiance_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h
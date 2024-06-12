import torch
import torch.nn.functional as F
import numpy as np
import mcubes
from models.visibility_tracer import VisibilityTracing
from models.ray_tracing import RayTracing
from models.general_utils import device
from models.polarization import stokes_fac_from_normal
import torch
import numpy as np

def project_points_to_image(points, intrinsics, extrinsics):
    points_homogeneous = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=-1)
    points_cam = torch.mm(extrinsics, points_homogeneous.T).T
    points_2d = torch.mm(intrinsics, points_cam[:, :3].T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:]
    return points_2d

def extract_fields_within_mask(bound_min, bound_max, resolution, query_func, mask, intrinsics, extrinsics):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    for mask_, extrinsics_ in zip(mask,extrinsics):
                        mask_ = torch.from_numpy(mask_).to(device)
                        extrinsics_ = extrinsics_.to(device)
                        projected_pts = project_points_to_image(pts, intrinsics, extrinsics_)
                        
                        valid_indices = (projected_pts[:, 0] >= 0) & (projected_pts[:, 0] < mask_.shape[1]) & \
                                        (projected_pts[:, 1] >= 0) & (projected_pts[:, 1] < mask_.shape[0])
                        valid_pts = projected_pts[valid_indices]
                        valid_3d_indices = torch.where(valid_indices)[0]
                        
                        if valid_pts.shape[0] == 0:
                            continue

                        valid_mask_values = mask_[valid_pts[:, 1].long(), valid_pts[:, 0].long()]
                        in_mask_indices = valid_3d_indices[valid_mask_values != 0]

                        if in_mask_indices.shape[0] == 0:
                            continue

                        val = query_func(pts[in_mask_indices]).detach().cpu().numpy()

                        idxs = (in_mask_indices // (len(ys) * len(zs))).detach().cpu().numpy()
                        idys = ((in_mask_indices % (len(ys) * len(zs))) // len(zs)).detach().cpu().numpy()
                        idzs = (in_mask_indices % len(zs)).detach().cpu().numpy()

                        u[xi * N + idxs, yi * N + idys, zi * N + idzs] = val.squeeze()
    return u



def extract_geometry(bound_min, bound_max, resolution, threshold, mask, intrinsics, extrinsics, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields_within_mask(bound_min, bound_max, resolution, query_func,mask, intrinsics, extrinsics)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 rough_network,
                 specular_network,
                 conf,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.specular_network = specular_network
        self.rough_network = rough_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.ray_tracer = RayTracing(**conf.get_config('model.ray_tracer'))
        self.visible_ray_tracer = VisibilityTracing(**conf.get_config('model.visibility_ray_tracer'))



    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    object_mask,idx,
                    dataset,
                    training,
                    only_diffuse,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    rough_network,
                    specular_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        refine = self.mvas_net(pts,object_mask,idx,sdf_network,device,dataset,training)
        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        sampled_color = color_network(pts, feature_vector).reshape(batch_size, n_samples, 3)
        rough_color = 0
        specular_color = 0
        polarized = True
        normal_vecs = F.normalize(gradients, dim=-1)

        if polarized: 
            stokes_diff_fac_i, stokes_spec_fac_i, stokes_spec_fac0_i = stokes_fac_from_normal(rays_o[...,None],rays_d[...,None,:],
                                                                              normal_vecs.reshape(batch_size,n_samples,3),
                                                                              ret_spec=True,
                                                                              clip_spec=True) 
            sampled_color = (sampled_color[...,None,:,None] * stokes_diff_fac_i).sum(-3)
            if not only_diffuse:
                roughness = rough_network.forward(pts, feature_vector)
                refl_vecs = dirs - (2 * torch.unsqueeze(torch.sum(dirs * normal_vecs, 1), 1) * normal_vecs)
                specular_color = specular_network.forward(refl_vecs, roughness).reshape(batch_size, n_samples, 3)
                specular_color = (specular_color[...,None,:,None] * stokes_spec_fac_i).sum(-3)
                rough_color = roughness.reshape(batch_size, n_samples, 1)

            
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)     
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None,None]).sum(dim=1)
        specular_color = (specular_color * weights[:, :, None,None]).sum(dim=1)
        rough_color =  (rough_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            "tangent_vectors_all_view": refine['tangent_vectors_all_view'],
            "tangent_vectors_all_view_half_pi": refine['tangent_vectors_all_view_half_pi'],
            "visibility_mask": refine['visibility_mask'],
            "surface_mask": refine['surface_mask'],
            "sdf_output":refine['sdf_output'],
            'network_object_mask': refine['network_object_mask'],
            'grad_theta': refine['grad_theta'],
            'specular_color': specular_color,
            'rough_color': rough_color,
            'color_all_view': refine['color_all_view'],
            'surface_normal': refine['surface_normal'],
        }
    
    def mvas_net(self,pts,mask,idx,sdf_network,device,dataset,training):
        if training:
            points, object_mask, idx = pts, mask, idx
            num_pixels, _ = pts.shape
            
            surface_mask = object_mask[...,None].repeat(1,128).reshape(-1)
            idx = idx[...,None].repeat(1,128).reshape(-1)
            sdf_output= self.sdf_network(points)[...,:1]
            
            eik_bounding_box = 1
            n_eik_points = num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).to(device)
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            grad_theta = sdf_network.gradient(eikonal_points)
            with torch.no_grad():
                visibility_mask = self.visible_ray_tracer(sdf=lambda x: sdf_network(x)[..., 0],
                                                          unique_camera_centers=dataset.unique_camera_centers.to(
                                                              device),
                                                          points=points[surface_mask])  # (num_points, num_cams)

                num_vis_points = visibility_mask.shape[0]
                visibility_mask[torch.arange(num_vis_points), idx[surface_mask].long()] = 1
                assert torch.all(visibility_mask.sum(-1) > 0)
                points_homo = torch.cat(
                    (points[surface_mask], torch.ones((surface_mask.sum(), 1), dtype=float, device=device)), -1).float()
                # project points onto all image planes
                # (num_cams, 3, 4) x (4, num_points)->  (num_cams, 3, num_points)
                pixel_coordinates_homo = torch.einsum("ijk, kp->ijp", dataset.projection_matrices.to(device),
                                                      points_homo.T).cpu().detach().numpy()
                pixel_coordinates_xx = (pixel_coordinates_homo[:, 0, :] / (
                            pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)
                pixel_coordinates_yy = (pixel_coordinates_homo[:, 1, :] / (
                            pixel_coordinates_homo[:, -1, :] + 1e-9)).T  # (num_points, num_cams)

                # opencv convention to numpy axis convention
                #  (top left) ----> x    =>  (top left) ---> axis 1
                #    |                           |
                #    |                           |
                #    |                           |
                #    y                         axis 0
                index_axis0 = np.round(pixel_coordinates_yy)  # (num_points, num_cams)
                index_axis1 = np.round(pixel_coordinates_xx)  # (num_points, num_cams)
                index_axis0 = np.clip(index_axis0, int(0), int(dataset.img_height - 1)).astype(
                    np.uint)  # (num_points, num_cams)
                index_axis1 = np.clip(index_axis1, int(0), int(dataset.img_width - 1)).astype(np.uint)

                num_cams = index_axis0.shape[1]
                tangent_vectors_all_view_list = []
                tangent_vectors_half_pi_all_view_list = []
                color_all_view_list = []
                for cam_idx in range(num_cams):
                    azimuth_angles = dataset.azimuth_map_all_view[cam_idx,
                                                                  index_axis0[:, cam_idx],
                                                                  index_axis1[:, cam_idx]]  # (num_surface_points)
                    R_list = dataset.W2C_list[cam_idx]
                    r1 = R_list[0, :3]
                    r2 = R_list[1, :3]
                    tangent_vectors_all_view_list.append(
                        r1 * np.sin(azimuth_angles[:, None]) - r2 * np.cos(azimuth_angles[:, None]))
                    tangent_vectors_half_pi_all_view_list.append(r1 * np.sin(azimuth_angles[:, None] + np.pi / 2) -
                                                                 r2 * np.cos(azimuth_angles[:, None] + np.pi / 2))
                    
                    color = torch.from_numpy(dataset.color_all_view[cam_idx,index_axis0[:, cam_idx],index_axis1[:, cam_idx]])
                    color_all_view_list.append(color)

                tangent_vectors_all_view = torch.stack(tangent_vectors_all_view_list, dim=1).to(
                    device)  # (num_points, num_cams, 3)
                tangent_vectors_half_pi_all_view = torch.stack(tangent_vectors_half_pi_all_view_list, dim=1).to(
                    device)  # (num_points, num_cams, 3)
                color_all_view = torch.stack(color_all_view_list,dim=1).to(device)

            output = {
                "tangent_vectors_all_view": tangent_vectors_all_view,
                "tangent_vectors_all_view_half_pi": tangent_vectors_half_pi_all_view,
                "visibility_mask": visibility_mask,
                "surface_mask": surface_mask,
                "sdf_output": sdf_output,
                'network_object_mask': 0,
                'grad_theta': grad_theta,
                'color_all_view': color_all_view,
                'surface_normal': 0
            }

        else:
            output = {
                "tangent_vectors_all_view": 0,
                "tangent_vectors_all_view_half_pi": 0,
                "visibility_mask": 0,
                "surface_mask": 0,
                "sdf_output": 0,
                'network_object_mask': 0,
                'grad_theta': 0,
                'color_all_view': 0,
                'surface_normal': 0
            }

        return output

    def render(self, rays_o, rays_d, mask,idx,dataset ,near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0,training= True, only_diffuse = True):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    mask,idx,
                                    dataset,
                                    training,
                                    only_diffuse,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    self.rough_network,
                                    self.specular_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            "tangent_vectors_all_view": ret_fine['tangent_vectors_all_view'],
            "tangent_vectors_all_view_half_pi": ret_fine['tangent_vectors_all_view_half_pi'],
            "visibility_mask": ret_fine['visibility_mask'],
            "surface_mask": ret_fine['surface_mask'],
            "sdf_output": ret_fine['sdf_output'],
            'network_object_mask': ret_fine['network_object_mask'],
            'grad_theta': ret_fine['grad_theta'],
            'specular_color': ret_fine['specular_color'],
            'rough_color': ret_fine['rough_color'],
            'color_all_view': ret_fine['color_all_view'],
            'surface_normal': ret_fine['surface_normal'],
        }

    def extract_geometry(self, bound_min, bound_max, resolution, mask, intrinsics, extrinsics,threshold=0.0,):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold, mask = mask, intrinsics = torch.from_numpy(intrinsics).to(device), extrinsics = extrinsics,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))

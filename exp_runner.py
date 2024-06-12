import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import *
import time
from pyhocon import ConfigFactory
from models.fields import SingleVarianceNetwork, NeRF, RoughNet, IntEnvMapNet, ImplicitSurface, DiffuseNet
from models.renderer import NeuSRenderer
from models.general_utils import device, get_class
from torch.utils.data import DataLoader
from  models.dataio.data_utils import *
from glob import glob
class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', select_model=None):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        with open(self.conf_path, 'r') as f:
            conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        self.conf = ConfigFactory.parse_string(conf_text)
        self.base_exp_dir_root = self.conf['general.base_exp_dir']
        self.now_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        self.base_exp_dir = os.path.join(self.base_exp_dir_root, self.now_time)
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.iter_step = 0
        
        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.train_resolution_level = self.conf.get_int('train.train_resolution_level')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.azimuth_weight = self.conf.get_float('train.azimuth_weight')
        self.use_half_pi_TSC_loss = self.conf.get_bool('train.use_half_pi_TSC_loss')
        self.silhouette_weight = self.conf.get_float('train.silhouette_weight')
        self.alpha = self.conf.get_float('train.alpha')
        self.mode = mode
        self.model_list = []
        self.writer = None
        self.data_type = self.conf.get_string('dataset.data_type')
        self.obj_name = case
        self.data_root = self.conf.get_string('dataset.data_root')
        self.train_dataset = get_class(self.conf["train"]["dataset_class"])(obj_name=self.obj_name,
                                                                            downscale=self.train_resolution_level,
                                                                            data_type = self.data_type,
                                                                            data_dir = self.data_root,
                                                                            batch_size=self.batch_size,
                                                                            use_pol = True,
                                                                            training=True,
                                                                            debug_mode = False,
                                                                            exclude_views = []  )
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=512, shuffle=True, pin_memory=True, generator=torch.Generator(device = 'cuda'))
        self.test_dataset = get_class(self.conf["train"]["dataset_class"])(obj_name=self.obj_name,
                                                                            downscale=self.validate_resolution_level,
                                                                            data_type = self.data_type,
                                                                            data_dir = self.data_root,
                                                                            batch_size=self.batch_size,
                                                                            use_pol = True,
                                                                            training=False,
                                                                            debug_mode = False,
                                                                            exclude_views = []  )
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, pin_memory=True, generator=torch.Generator(device = 'cuda'))
        self.test_list = enumerate(self.test_dataloader)
        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = ImplicitSurface().to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = DiffuseNet().to(self.device)
        self.rough_network = RoughNet(**self.conf['model.rough_network']).to(self.device)
        self.specular_network = IntEnvMapNet(**self.conf['model.specular_network']).to(self.device)

        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        params_to_train += list(self.rough_network.parameters())
        params_to_train += list(self.specular_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     self.rough_network,
                                     self.specular_network,
                                     self.conf,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        if select_model is not None and mode != 'train':
            if select_model == 'latest':
                model_path_root = os.path.join(self.base_exp_dir_root, 'checkpoints')
                model_list = os.listdir(model_path_root)
                model_list.sort()
                select_model_file = os.path.join(model_path_root, model_list[-1])
                model_list = []
                for model_name in os.listdir(select_model_file):
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                select_model_file = os.path.join(select_model_file, model_list[-1])
            else:
                select_model_file = select_model
            self.load_checkpoint(select_model_file)


        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        image_perm = self.get_image_perm()
        only_diffuse = True
        use_pol = True
        while self.iter_step < self.end_iter:
            pbar_batch = tqdm(enumerate(self.train_dataloader),total=len(self.train_dataloader))
            for idx_,model_input in pbar_batch:
                self.iter_step += 1

                rays_o, rays_d, true_rgb, mask, idx, s0_gt, s1_gt, s2_gt= model_input["camera_center"].squeeze().to(device),model_input["view_direction"].squeeze().to(device)\
                    ,model_input["img"].squeeze().to(device),model_input["object_mask"].squeeze()[...,None].to(device), model_input["view_idx"].squeeze().to(device)\
                    ,model_input["s0"].squeeze().to(device), model_input["s1"].squeeze().to(device), model_input["s2"].squeeze().to(device)
                self.a_mask = mask.squeeze()

                near, far = near_far_from_sphere(rays_o, rays_d)

                background_rgb = None
                if self.use_white_bkgd:
                    background_rgb = torch.ones([1, 3])

                if self.mask_weight > 0.0:
                    mask = (mask > 0.5).float()
                else:
                    mask = torch.ones_like(mask).float()
                tr = False
                if self.iter_step>= 1000:
                    tr = True
                mask_sum = mask.sum() + 1e-5
                if self.iter_step>=1000:
                    only_diffuse = False
                render_out = self.renderer.render(rays_o, rays_d, self.a_mask, idx, self.train_dataset, near, far,
                                                background_rgb=background_rgb,
                                                cos_anneal_ratio=self.get_cos_anneal_ratio(),training=tr,only_diffuse=only_diffuse)
                color_diffuse = render_out['color_fine'][...,0]
                color_specular = render_out['specular_color'][...,0]
                color_fine_loss = F.l1_loss((color_diffuse+color_specular) * mask, s0_gt * mask,reduction='sum')/ mask_sum

                if use_pol and self.iter_step>=1000:
                    color_diffuse_s1 = render_out['color_fine'][...,1]
                    color_diffuse_s2 = render_out['color_fine'][...,2]
                    color_specular_s1 = render_out['specular_color'][...,1]
                    color_specular_s2 = render_out['specular_color'][...,2]
                    s1_loss = F.l1_loss((color_diffuse_s1+color_specular_s1) * mask, s1_gt * mask,reduction='sum')/ mask_sum
                    s2_loss = F.l1_loss((color_diffuse_s2+color_specular_s2) * mask, s2_gt * mask,reduction='sum')/ mask_sum
                else:
                    s1_loss, s2_loss = 0,0

                s_val = render_out['s_val']
                gradient_error = render_out['gradient_error']
                weight_sum = render_out['weight_sum']
                azimuth_loss=0

                psnr = 20.0 * torch.log10(1.0 / (((color_diffuse + color_specular - s0_gt)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

                eikonal_loss = gradient_error
                if self.iter_step>=1000:
                    self.azimuth_weight = 1
                    azimuth_loss = self.get_azimuth_loss(render_out,model_input)
                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-7, 1.0 - 1e-7), mask)

                loss = color_fine_loss +\
                    eikonal_loss * self.igr_weight +\
                    mask_loss * self.mask_weight+\
                    s1_loss+s2_loss+\
                    azimuth_loss * self.azimuth_weight

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
                self.writer.add_scalar('Loss/maskloss', mask_loss, self.iter_step)


                if self.iter_step % self.report_freq == 0:
                    print(self.base_exp_dir)
                    print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))
                    print('rgbloss={} azimuthloss={} maskloss={} eikloss={}'.format(color_fine_loss, azimuth_loss, mask_loss, eikonal_loss))

                if self.iter_step % self.save_freq == 0:
                    self.save_checkpoint()
                if self.iter_step % self.val_freq == 0:
                    self.validate_image(only_diffuse,self.a_mask)

                self.update_learning_rate()

                if self.iter_step % len(image_perm) == 0:
                    image_perm = self.get_image_perm()


    def get_eikonal_loss(self,grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss


    def get_azimuth_loss(self,outputs,inputs):

        gradients = outputs['gradients']
        weights = outputs['weights'].reshape(-1, 1)
        inside_sphere = outputs['inside_sphere']
        surface_mask = outputs['surface_mask']
        object_mask = inputs['object_mask'] 
        object_mask = object_mask.squeeze()
        normals = gradients 
                
        normals = normals * inside_sphere[..., None] 
        normals = normals.reshape(-1, 3) 

        TSC_loss =self.get_half_pi_tangent_space_consistency_loss(normals,
                                                                        outputs["tangent_vectors_all_view"],
                                                                        outputs[
                                                                            "tangent_vectors_all_view_half_pi"],
                                                                        outputs["visibility_mask"],
                                                                        weights,surface_mask, object_mask)
        

        return TSC_loss


    def get_half_pi_tangent_space_consistency_loss(self, normals, tangents_all_view, tangents_all_view_pi2,
                                                   visibility_mask, weights, surface_mask,object_mask):

        normals = F.normalize(normals,dim=-1)
        weights = weights[surface_mask] 
        normals = normals[surface_mask]
        not_nan_mask = ~torch.isnan(tangents_all_view.sum(-1))
        visibility_mask = visibility_mask & not_nan_mask  
        tangents_all_view[torch.isnan(tangents_all_view)] = 1  
        tangents_all_view_pi2[torch.isnan(tangents_all_view_pi2)] = 1
        num_visible_views = visibility_mask.sum(-1)

        loss_1= ((normals.unsqueeze(1) * tangents_all_view).sum(-1)) ** 2  
        loss_1 = loss_1 * weights

        loss_2 = ((normals.unsqueeze(1) * tangents_all_view_pi2).sum(-1)) ** 2
        loss_2 = loss_2 * weights

        loss_1 = loss_1 * visibility_mask
        loss_2 = loss_2* visibility_mask


        visible_view_mask = num_visible_views > 0
        loss = loss_1 * loss_2

        loss = loss[visible_view_mask].sum(-1) / num_visible_views[visible_view_mask]
        loss = loss.sum() / float(visible_view_mask.shape[0])
        return loss
    
    def get_image_perm(self):
        return torch.arange(0, self.train_dataset.num_views, 1)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(checkpoint_name, map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.rough_network.load_state_dict(checkpoint['rough_network'])
        self.specular_network.load_state_dict(checkpoint['specular_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'rough_network': self.rough_network.state_dict(),
            'specular_network': self.specular_network.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir_root, 'checkpoints', self.now_time), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir_root, 'checkpoints', self.now_time, 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self,only_diffuse=False,a_mask=0,idx=-1, resolution_level=-1):

        training = False

        idx,input = next(iter(self.test_list))

        if idx == len(self.test_dataset)-1:
            self.test_list = enumerate(self.test_dataloader)
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        
        rays_o, rays_d , mask_all= input["camera_center"].squeeze().to(device),input["view_direction"].squeeze().to(device),input["mask"].reshape(-1,1).to(device)
        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))
        H, W = self.test_dataset.img_height, self.test_dataset.img_width
        rays_o = rays_o.split(self.batch_size)
        rays_d = rays_d.split(self.batch_size)
        mask = mask_all.split(self.batch_size)

        out_rgb_fine = []
        out_normal_fine = []
        out_specular_fine =[]
        out_albedo_fine = []
        out_roughness_fine = []
        for rays_o_batch, rays_d_batch,mask in zip(rays_o, rays_d,mask):
            near, far = near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,a_mask,idx,self.test_dataset,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb,training=training,only_diffuse=only_diffuse)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'][...,0].detach().cpu().numpy()+render_out['specular_color'][...,0].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                normals = normals.sum(dim=1) * mask
                normals = F.normalize(normals,dim=-1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            if feasible('specular_color'):
                out_specular_fine.append(render_out['specular_color'][...,0].detach().cpu().numpy())
            if feasible('color_fine'):
                out_albedo_fine.append(render_out['color_fine'][...,0].detach().cpu().numpy())
            if feasible('rough_color'):
                out_roughness_fine.append(render_out['rough_color'][...,0].detach().cpu().numpy())
            del render_out


        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1])).clip(0, 1)**(1/2.2)*255

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0).reshape([H, W, 3, -1])
            w2c = self.test_dataset.W2C_list[idx, :3, :3]
            # transform w2c to tensor
            w2c = torch.tensor(w2c, dtype=torch.float32).to(device)
            rot = np.linalg.inv((w2c.permute(1,0)).detach().cpu().numpy())
            normal_img = normal_img[:, :, None]
            normal_img = np.matmul(rot[None, :, :], normal_img).reshape([H, W, 3, -1]) 
            normal = normal_img.copy()
            if self.data_type == 'pandora' or self.data_type == 'ours_real':
                normal[...,0,:] = -normal_img[...,2,:]
                normal[...,1,:] = -normal_img[...,1,:]
                normal[...,2,:] = normal_img[...,0,:]
            elif self.data_type == 'ours_synthetic':
                normal[...,0,:] = -normal_img[...,2,:]
                normal[...,1,:] = normal_img[...,1,:]
                normal[...,2,:] = -normal_img[...,0,:]
            normal_img = ((normal+1)/2).clip(0,1)*255

        roughness_fine = None
        if len(out_roughness_fine) > 0:
            roughness_fine = (np.concatenate(out_roughness_fine, axis=0).reshape([H, W, 1, -1]))
            roughness_fine = roughness_fine/roughness_fine.max()
            roughness_fine = roughness_fine.clip(0,1)*255
        specular_fine = None
        if len(out_specular_fine) > 0:
            specular_fine = (np.concatenate(out_specular_fine, axis=0).reshape([H, W, 3, -1])).clip(0, 1)**(1/2.2)*255
        albedo_fine = None
        if len(out_albedo_fine) > 0:
            albedo_fine = (np.concatenate(out_albedo_fine, axis=0).reshape([H, W, 3, -1])).clip(0, 1)**(1/2.2)*255


        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'specular'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'albedo'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'roughness'), exist_ok=True)
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i][...,::-1],
                                           self.test_dataset.image_at(idx, resolution_level=resolution_level)]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])
            if len(out_specular_fine) > 0:
                 cv.imwrite(os.path.join(self.base_exp_dir,
                                        'specular',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           specular_fine[..., i][...,::-1])
            if len(out_albedo_fine) > 0:
                 cv.imwrite(os.path.join(self.base_exp_dir,
                                        'albedo',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           albedo_fine[..., i][...,::-1])
            if len(out_roughness_fine) > 0:
                    cv.imwrite(os.path.join(self.base_exp_dir,
                                            'roughness',
                                            '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                            roughness_fine[..., i][...,::-1])


    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.test_dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.test_dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, mask = self.test_dataset.mask_list,\
                                            intrinsics=self.test_dataset.K[:3,:3], extrinsics=self.test_dataset.W2C_list, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.test_dataset.scale_mats_np[0][0, 0] + self.test_dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')



if __name__ == '__main__':
    torch.random.manual_seed(0)
    np.random.seed(0)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/wmask_ours_synthetic.conf')
    parser.add_argument('--mode', type=str, default='validate_mesh')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--select_model',type=str,default='latest')
    parser.add_argument('--case', type=str, default='25_6')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.select_model)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
        for i in range(len(runner.test_dataset)):
            runner.validate_image()


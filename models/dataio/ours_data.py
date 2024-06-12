import os
from glob import glob
import cv2
import numpy as np
import torch
from skimage.transform import rescale
from torch.utils.data import Dataset
from models.general_utils import device, boundary_expansion_mask
from models.dataio.data_utils import normalize_camera
import imageio
import skimage

def load_rgb(path, downscale=1):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)
    if downscale != 1:
        img = rescale(img, 1./downscale, anti_aliasing=False, multichannel=True)

    return img

def load_rgb_npy(path,downscale=1):
    img = np.load(path)
    if downscale != 1:
        img = rescale(img, 1./downscale, anti_aliasing=False, multichannel=True)
    return img


class OurDataloader(Dataset):
    def __init__(self, data_dir, obj_name, batch_size, use_pol, training = True, downscale=None, data_type = 'ours',
                 exclude_views=[], debug_mode=False):
        self.training = training
        self.data_dir = os.path.join(data_dir, obj_name)
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png'))+glob(os.path.join(self.data_dir, 'image/*.jpg')))

        self.downscale = downscale
        self.exclude_views = exclude_views
        self.batch_size = batch_size
        # ==============================================load camera parameters =========================================
        self.poses_from_colmap()
        # ==============================================================================================================
        camera_center_repeated_list = []  # in world coordinates
        mask_vectorized_list = []
        tangents_list = []
        tangents_half_pi_list = []
        self.tangents_all_view = []
        view_direction_list = []  # in world coordinates
        self.view_direction_all_view = []
        self.mask_all_view = []
        self.object_mask_all_view = []
        self.expanded_mask_all_view = []
        self.input_azimuth_angle_map_vis_all_view = []
        self.input_azimuth_angle_map_pi2_vis_all_view = []
        self.dolp_all_view = []
        self.dolp_all_view_vis = []
        self.tangents_map_list = []
        self.tangents_map_list_pi2 = []
        self.tangents_pi2_all_view = []

        view_idx_list = []
        self.camera_center_all_view = []
        mask_list = []
        azimuth_list = []
        azimuth_map_all_view = []
        color_all_view = []
        num_pixel_in_each_view = []
        img_gt_list = []
        s0_gt_list = []
        s1_gt_list = []
        s2_gt_list = []
        self.use_pol = use_pol
        view_count = -1
        for view_idx in range(1, self.num_views + 1):
            view_count += 1
            if view_idx in exclude_views:
                view_count -= 1
                print(f"View {view_idx} is excluded!")
                continue
            else:
                print(f"Processing view {view_idx}...")
            
            view_mask = cv2.imread(os.path.join(self.data_dir, "input_azimuth_maps",'{:04d}.png'.format(view_idx-1)),
                                   -1)[..., -1].astype(bool)
            img = cv2.imread(os.path.join(self.data_dir, "image",'{:04d}.png'.format(view_idx-1)))[...,:3]
            img =img.astype(np.float32)/256

            if downscale is not None:
                view_mask = rescale(view_mask, 1. / downscale, anti_aliasing=False)
                img = rescale(img, 1. / downscale,anti_aliasing=False,multichannel=True)
            mask_list.append(view_mask.astype(bool))

            expanded_mask = boundary_expansion_mask(view_mask)

            for _ in range(30):
                expanded_mask = boundary_expansion_mask(expanded_mask)
            # expanded_mask = np.ones_like(view_mask).astype(bool)
            mask_vectorized_list.append(view_mask[expanded_mask])
            num_pixel_in_each_view.append(np.sum(expanded_mask))
            view_idx_list.append(np.full(shape=(np.sum(expanded_mask)), fill_value=view_count))

            if use_pol:
                if data_type == 'pandora':
                ################  only for pandora
                    s0 = 0.5*load_rgb(f'{self.data_dir}/images_stokes/{int(view_idx):02d}_s0.hdr', downscale) 
                    s0p1 = 0.5*load_rgb(f'{self.data_dir}/images_stokes/{int(view_idx):02d}_s0p1.hdr', downscale)
                    s0p2 = 0.5*load_rgb(f'{self.data_dir}/images_stokes/{int(view_idx):02d}_s0p2.hdr', downscale)
                    s1 = s0p1-s0
                    s2 = s0p2-s0
                # ################### for synthetic data and our real data
                elif data_type == 'ours_real' or data_type == 'ours_synthetic':
                    s0 =load_rgb_npy(f'{self.data_dir}/s0/{int(view_idx-1):04d}.npy', downscale) *0.5
                    s1 =load_rgb_npy(f'{self.data_dir}/s1/{int(view_idx-1):04d}.npy', downscale) *0.5
                    s2 =load_rgb_npy(f'{self.data_dir}/s2/{int(view_idx-1):04d}.npy', downscale) *0.5
                else:
                    raise NotImplementedError
                #######################################
                color_all_view.append(s0)

                s0_gt_list.append(s0[expanded_mask])
                s1_gt_list.append(s1[expanded_mask])
                s2_gt_list.append(s2[expanded_mask])


            # ====================================load azimuth angle maps================================================
            # azimuth_map = cv2.imread(os.path.join(self.data_dir, "input_azimuth_maps", '{:04d}.png'.format(view_idx-1)), -1)[
            #     ..., 0]
            azimuth_map = cv2.imread(os.path.join(self.data_dir, "input_azimuth_maps", '{:04d}.png'.format(view_idx-1)), -1)[
                ..., 0]
            azimuth_map = np.pi * azimuth_map / 65535

            if self.downscale is not None:
                azimuth_map = cv2.resize(azimuth_map,
                                         dsize=None, fx=1 / downscale, fy=1 / downscale,
                                         interpolation=cv2.INTER_NEAREST)

            a = azimuth_map[expanded_mask]

            azimuth_list.append(a)
            azimuth_map_all_view.append(azimuth_map)

            img_gt_mask = img[expanded_mask]
            img_gt_list.append(img_gt_mask)


            W2C = self.W2C_list[view_count]
            C2W = np.linalg.inv(W2C)

            R = W2C[:3, :3]
            t = W2C[:3, 3]

            r1 = R[0] # 3，
            r2 = R[1] # 3，
            tangents = r1[None, :] * np.sin(a)[:, None] - r2[None, :] * np.cos(a)[:, None]
            tangents_pi2 = r1[None, :] * np.sin(a + np.pi / 2)[:, None] - r2[None, :] * np.cos(a + np.pi / 2)[:, None]

            tangents_list.append(tangents)
            tangents_half_pi_list.append(tangents_pi2)

            camera_center = - R.T @ t  # in world coordinate
            camera_center_scaled=camera_center
            # camera_center_scaled = (
            #                                    camera_center - self.normalized_coordinate_center) / self.normalized_coordinate_scale
            
            if self.training:
                camera_center_repeated_list.append(np.tile(camera_center_scaled.T, (np.sum(expanded_mask), 1)))
            else:
                camera_center_repeated_list.append(np.tile(camera_center_scaled.T, (self.img_width*self.img_height, 1)))
            self.camera_center_all_view.append(
                torch.from_numpy(np.tile(camera_center_scaled.T, (np.sum(expanded_mask), 1))).float())

            # ======================================compute ray directions===============================================
            # xx -> right yy -> bottom
            xx, yy = np.meshgrid(range(self.img_width), range(self.img_height))
            if self.training:
                uv_homo = np.stack((xx[expanded_mask], yy[expanded_mask], np.ones_like(xx[expanded_mask])), axis=-1)
            else:
                uv_homo = np.stack((xx.reshape((-1)), yy.reshape((-1)), np.ones_like(xx.reshape((-1)))), axis=-1)
            view_direction = (C2W[:3, :3] @ np.linalg.inv(self.K[:3, :3]) @ uv_homo.T).T
            view_direction = view_direction / np.linalg.norm(view_direction, axis=-1, keepdims=True)
            view_direction_list.append(view_direction)

        if debug_mode:
            from visual_hull_check import visual_hull_creation
            visual_hull_creation(mask_list, self.P_list)



        # training data
        self.mask_vectorized = torch.from_numpy(np.concatenate(mask_vectorized_list, 0)).bool()
        self.view_direction = torch.from_numpy(np.concatenate(view_direction_list, 0)).float()
        self.camera_center = torch.from_numpy(np.concatenate(camera_center_repeated_list, 0)).float()
        self.view_idx = torch.from_numpy(np.concatenate(view_idx_list)).int()
        self.img_gt = torch.from_numpy(np.concatenate(img_gt_list,0)).float()
        if self.use_pol:
            self.s0_gt = torch.from_numpy(np.concatenate(s0_gt_list, 0)).float()
            self.s1_gt = torch.from_numpy(np.concatenate(s1_gt_list, 0)).float()
            self.s2_gt = torch.from_numpy(np.concatenate(s2_gt_list, 0)).float()

        self.view_direction_val =view_direction_list
        self.camera_center_val =camera_center_repeated_list
        self.mask_list = mask_list

        self.unique_camera_centers = torch.from_numpy(
            np.squeeze(np.array(self.unique_camera_center_list))).float()  # (num_cams, 3)
        self.projection_matrices = torch.from_numpy(np.array(self.P_list)).float() # (num_cams, 3, 4)
        self.azimuth_map_all_view = np.array(azimuth_map_all_view)  # (num_cams, img_height, img_widht)
        self.color_all_view = np.array(color_all_view)
        self.W2C_list = torch.from_numpy(np.array(self.W2C_list)).float()

    def poses_from_colmap(self):
        img_list = os.listdir(os.path.join(self.data_dir,'image'))
        self.num_views = len(img_list)
        realdir = self.data_dir
        cameras_dict_file = os.path.join(realdir,"cameras.npz")
        camera_dict = np.load(cameras_dict_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.num_views)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.num_views)]
        img = cv2.imread(os.path.join(self.data_dir,'image',img_list[0]))
        h,w,_ = img.shape
        self.img_height = h
        self.img_width = w
        del img

        P0 = world_mats[0] @ scale_mats[0]
        P0 = P0[:3,:4]
        out0 = cv2.decomposeProjectionMatrix(P0)
        K = out0[0]
        self.K = K/K[2,2]
        if self.downscale is not None:
            self.K /= self.downscale
            self.K[2,2] =1
            self.img_height = int(np.round(self.img_height/ self.downscale))
            self.img_width = int(np.round(self.img_width / self.downscale))

        self.W2C_list = []
        self.C2W_list = []
        self.W2C_scaled_list = []
        self.C2W_scaled_list = []
        self.P_list = []
        self.unique_camera_center_list = []

        R_list = []
        t_list = []
        
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            out = cv2.decomposeProjectionMatrix(P)
            R = out[1]
            c= out[2]
            c= (c[:3] / c[3])
            c = c.copy()
            t = -R @ c
            R_list.append(R)
            t_list.append(t)

        self.normalized_coordinate_center, self.normalized_coordinate_scale = normalize_camera(R_list, t_list,
                                                                                               camera2object_ratio=2)
        print(f"camera centers are shifted by {self.normalized_coordinate_center} "
              f"and scaled by {self.normalized_coordinate_scale}!")

        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            out = cv2.decomposeProjectionMatrix(P)
            R = out[1]
            c = out[2]
            c = (c[:3] / c[3])
            t = -R @ c
            camera_center =  c  # in world coordinate
            self.unique_camera_center_list.append(camera_center)
            W2C = np.zeros((4, 4))
            W2C[3, 3] = 1
            W2C[:3, :3] = R
            W2C[:3, [3]] = t
            self.W2C_list.append(W2C)
            C2W = np.linalg.inv(W2C)
            self.C2W_list.append(C2W)
            W2C_scaled = np.zeros((4, 4), float)
            W2C_scaled[:3, :3] = self.normalized_coordinate_scale * R
            W2C_scaled[:3, 3] = np.squeeze(t) + R @ self.normalized_coordinate_center
            W2C_scaled[-1, -1] = 1
            C2W_scaled = np.linalg.inv(W2C_scaled)
            self.W2C_scaled_list.append(W2C_scaled)
            self.C2W_scaled_list.append(C2W_scaled)
            P = self.K[:3, :3] @ W2C[
                                 :3]  
            self.P_list.append(P)

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])

        object_bbox_min = object_bbox_min[:, None]
        object_bbox_max = object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]


    def image_at(self, idx, resolution_level):
        img = cv2.imread(self.images_lis[idx])
        return (cv2.resize(img, (self.img_width, self.img_height))).clip(0, 255)

    def __getitem__(self, idx):
        if self.training:
            model_input = {
                "camera_center": self.camera_center[idx],
                "view_direction": self.view_direction[idx],
                "object_mask": self.mask_vectorized[idx],
                "view_idx": self.view_idx[idx],
                "img": self.img_gt[idx]
            }
            if self.use_pol:
                model_input['s0'] = self.s0_gt[idx]
                model_input['s1'] = self.s1_gt[idx]
                model_input['s2'] = self.s2_gt[idx]
        else:
            model_input = {
                "camera_center": torch.from_numpy(self.camera_center_val[idx]).float(),
                "view_direction": torch.from_numpy(self.view_direction_val[idx]).float(),
                "mask": torch.from_numpy(self.mask_list[idx]).bool()
            }
        return model_input

    def __len__(self):
        if self.training:
            return len(self.mask_vectorized)
        else:

            return self.num_views




if __name__ == '__main__':
    # sanity check
    OurDataloader(data_dir="../../data/", obj_name="a2_ceramic_owl_v2",
                      downscale=4,
                      debug_mode=True)

import torch
import numpy as np

def near_far_from_sphere(rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far


def normalize_camera(R_list, t_list, camera2object_ratio=10):
    A_camera_normalize = 0
    b_camera_normalize = 0
    camera_center_list = []
    for view_idx in range(len(R_list)):
        R = R_list[view_idx]
        t = t_list[view_idx]
        camera_center = - R.T @ t  # in world coordinate
        camera_center_list.append(camera_center)
        vi = R[2][:, None]  # the camera's principal axis in the world coordinates
        Vi = vi @ vi.T
        A_camera_normalize += np.eye(3) - Vi
        b_camera_normalize += camera_center.T @ (np.eye(3) - Vi)
    offset = np.linalg.lstsq(A_camera_normalize, np.squeeze(b_camera_normalize), rcond=None)[0]
    camera_center_dist_list = [np.sqrt(np.sum((np.squeeze(c) - offset) ** 2))
                               for c in camera_center_list]
    scale = np.max(camera_center_dist_list) / camera2object_ratio

    return offset, scale
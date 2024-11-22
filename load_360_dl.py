import os
import cv2
import torch
from torch.utils.data import Dataset
# from torchvision import transforms
import math
import numpy as np
import random
from run_nerf_helpers_new import get_rays_np_sp,get_rays_np
from run_nerf_helpers_new import get_rays_np_roll,get_rays_roll
class load_360(Dataset):
    def __init__(self, data_dir, W=6720,H=3360,mode="train",bds='sp',patch=True,transform=None,N_rand=1024):
        self.patch = patch
        self.root_dir = ""
        self.transform = transform
        self.img_fpath = ""
        self.patch_size = None
        self.poses = {}
        self.mode=mode
        self.datanum=0
        self.N_rand = N_rand

        if self.mode == "train":
            self.root_dir = os.path.join(data_dir+'/train/')
            self.img_fpath = os.path.join(self.root_dir+'images/')
        else :
            self.root_dir = os.path.join(data_dir+'/test/')
            self.img_fpath = os.path.join(self.root_dir+'images/')
        img_list = []
        with open(os.path.join(self.root_dir, 'poses.txt')) as f:
            for line in f.readlines():
                line = line.rstrip()
                line = line.split(" ")
                pose = self.transform_pose(line)
                img_list.append(line[0]+".png")
                self.poses[line[0]]=pose
        H,W,_ = cv2.imread(os.path.join(self.img_fpath, line[0]+".png")).shape
        self.W = W
        self.H = H
        self.image_files = []
        if self.patch and mode=="train":
            self.patch_size = [3,4]
            patch_n = self.patch_size[0]
            patch_m = self.patch_size[1]
            for f in img_list:
                for n in range(patch_n):
                    for m in range(patch_m):
                        self.datanum+=1
                        self.image_files.append([os.path.join(self.img_fpath+f),[n,m]])
        else:
           for f in img_list:
            self.datanum+=1
            self.image_files.append([os.path.join(self.img_fpath+f),[]])
            # self.image_files = [
            #     f for f in os.listdir(self.img_fpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            # ]
        if bds == 'sp':
            self.getray = get_rays_np_sp
        else:
            self.getray = get_rays_np_roll
    def rtn_hwf(self):
        return [self.H,self.W]
    def roll_rotation_matrix(self,angle_degrees):
        angle_radians = math.radians(angle_degrees)
        return np.array([
            [1, 0, 0],
            [0, math.cos(angle_radians), -math.sin(angle_radians)],
            [0, math.sin(angle_radians), math.cos(angle_radians)]
        ])

    def transform_pose(self,line):
        # 元の姿勢行列を作成
        # make_transform_pose
        transform_pose = np.zeros((4,4),np.float32)
        transform_pose[3,3] = 1.0
        # make_rotationMATRIX
        rotation_matrix = np.array([float(x) for x in line[1:10]]).reshape(3,3)
        translation = np.array([float(x) for x in line[10:13]]).reshape(3)
        # roll_rotation = roll_rotation_matrix(-270)
        # -90Degree_FIXed
        # rotation_matrix = roll_rotation @ rotation_matrix
        # translation = translation @ roll_rotation
        # Applyed RotationMatrix
        transform_pose[:3,:3] = rotation_matrix
        transform_pose[0:3,3] = translation.T
        return transform_pose

    def __len__(self):
        return len(self.image_files)
    def path_split(self, img):
        h, w, c = img.shape
        patch_h, patch_w = self.patch_size
        patches = []
        for i in range(0, h, h//patch_h):
            for j in range(0, w, w//patch_w):
                patch = img[i:i+h//patch_h, j:j+w//patch_w, :]
                patches.append(patch)
        patches = np.array(patches)
        patches = patches.reshape((4*3, w//patch_w, h//patch_h, c))
        return patches
    def random_patch(self,img,ray_o,ray_d):
        # num_elements = img.shape[0]
        img = img.reshape(-1, 3)
        ray_o = ray_o.reshape(-1, 3)
        ray_d = ray_d.reshape(-1, 3)
        indices = np.random.randint(0, img.shape[0], size=self.N_rand)
        if self.N_rand >= img.shape[1] * img.shape[0]:
            return img,ray_o,ray_d
        img = img[indices,:]
        ray_o = ray_o[indices]
        ray_d = ray_d[indices]
        return img,ray_o,ray_d
    def __getitem__(self, idx):
        img_path,patch_idx = self.image_files[idx]
        img = cv2.imread(img_path)/255.
        c2w = self.poses[img_path[-7:-4]]
        ray_o,ray_d =self.getray(H=img.shape[0],W=img.shape[1],c2w=c2w,K=None)
        if self.mode == 'train':
            if self.patch and  self.mode == 'train':
                patch_img = self.path_split(img)
                patch_ray_o= self.path_split(ray_o)
                patch_ray_d= self.path_split(ray_d)
                patch_img,patch_ray_o,patch_ray_d=self.random_patch(patch_img[patch_idx[0],patch_idx[1]],patch_ray_o[patch_idx[0],patch_idx[1]],patch_ray_d[patch_idx[0],patch_idx[1]])
                rtn_img =  patch_img
                rtn_ray_o =  patch_ray_o
                rtn_ray_d = patch_ray_d
            else :
                # patch_img = self.path_split(img)
                # patch_ray_o= self.path_split(ray_o)
                # patch_ray_d= self.path_split(ray_d)
                patch_img,patch_ray_o,patch_ray_d=self.random_patch(img,ray_o,ray_d)
                rtn_img =  patch_img
                rtn_ray_o =  patch_ray_o
                rtn_ray_d = patch_ray_d
        else :
            rtn_img = img
            rtn_ray_o = []
            rtn_ray_d = []
        c2w = torch.from_numpy(np.array(c2w,dtype=np.float32)).clone()
        rtn_img = torch.from_numpy(np.array(rtn_img.astype(np.float32),dtype=np.float32)).clone()
        rtn_ray_o = torch.from_numpy(np.array(rtn_ray_o,dtype=np.float32)).clone()
        rtn_ray_d = torch.from_numpy(np.array(rtn_ray_d,dtype=np.float32)).clone()
        return c2w,rtn_img,rtn_ray_o,rtn_ray_d
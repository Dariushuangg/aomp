# Author: Darius H. 2023/10/15

import cv2
import numpy as np
import os
import torch
import core.transforms as transforms
import torch.nn.functional as F

#!TODO feature caching
class ImageDatabase():
    def __init__(self, model, data_dir):
        self._img_feats = []
        self._model = model
        self._data_dir = data_dir
        
        mean, sd = self.compute_mean_variance() 
        self._MEAN = mean
        self._SD = sd
    
    @torch.no_grad()
    def extract_img_feat(self, image):
        image = self.preprocess_img(image)
        desc = self._model.extract_global_descriptor(image)
        
        if len(desc.shape) == 1:
            desc.unsqueeze_(0)
        
        return desc.detach().cpu()
    
    @torch.no_grad()
    def extract_all_img_feats(self):
        """
        Import the database images and calculate global feature.
        """
        folder_dir = os.path.join(os.path.dirname(__file__), self._data_dir)
        for filename in os.listdir(folder_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')): 
                filepath = os.path.join(folder_dir, filename)
                image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                feat = self.extract_img_feat(image)
                self._img_feats.append(feat)
                
        self._img_feats = F.normalize(self._img_feats, p=2, dim=1) # L2 normalization of the feature vectors
        self._img_feats = self._img_feats.cpu().numpy()
    
    def preprocess_img(self, im):
        """
        Take an (B, G, R) image stored in [H, W, C] format, convert into
        color normalized [C, H, W] format.
        """
        im = im.transpose([2, 0, 1]) 
        # [0, 255] -> [0, 1]
        im = im / 255.0
        # Color normalization
        im = transforms.color_norm(im, self._MEAN, self._SD)
        return im
    
    def compute_mean_variance(self):
        """
        Compute [0, 1] mean and sd for (B, G, R) image
        """
        sum_pixels = 0
        sum_pixels_squared = 0
        num_pixels = 0

        # Iterate over all images in the folder
        folder_dir = os.path.join(os.path.dirname(__file__), self._data_dir)
        for filename in os.listdir(folder_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')): 
                filepath = os.path.join(folder_dir, filename)

                image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"Error reading {filename}")
                    continue
                
                # Convert to float32 for computation
                image = image.astype(np.float32)
                image = image / 255

                # Update the sums
                sum_pixels += np.sum(image, axis=(0, 1))
                sum_pixels_squared += np.sum(image**2, axis=(0, 1))
                num_pixels += image.shape[0] * image.shape[1]

        # Compute mean and variance for each channel (B, G, R)
        mean = sum_pixels / num_pixels
        variance = (sum_pixels_squared / num_pixels) - (mean ** 2)
        
        return mean, variance

# Test 
# db = ImageDatabase([], 'resorts/forbidden_city') # Arrange

# Test - mean & var for dataset 
# db.compute_mean_variance()
# print(db._MEAN, db._SD)
# # assert db._MEAN == ? 


# Test - feature calculation
# db.extract_all_img_feats()
# print(db._img_feats.shape)
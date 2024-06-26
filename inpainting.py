import os
import numpy as np
from dps import DPS
from image_loader import FFHQ_dataloader
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import image_tools

def box_type(image, box_size=128, margin=16):
    """
    Box-type inpainting, as specified by DPS paper [1] and Chung et all 2022a [2].

    Args: 
     - box_size: we mask out a 'box_size'*'box_size' box region from the image, default 128 from DPS paper
     - margin: the location of the box is sampled uniformly within 'margin' # of pixel margins on each side

    Returns:
     - start_h, start_w: top left corner's height and width coordinates
    """
    h, w = image.shape[-2:]
    
    start_h = np.random.randint(margin, h - margin - box_size)
    start_w = np.random.randint(margin, w - margin - box_size)
    
    return start_h, start_w

def box_type_mask(image, start_h, start_w, box_size=128):
    """
    A operator for box_type inpainting mask.

    Args:
     - start_h, start_w: such that this operator can deterministically map the picture to the inpainted box version (when comparing measurement y to A(x0_hat))
     - box_size: default 128
    """
    mask = torch.ones_like(image)
    mask[:, :, start_h:start_h+box_size, start_w:start_w+box_size] = 0
    return image * mask

def random_type(image, mask_amount = 0.92):
    """
    Random type inpainting masking as specified by [1]

    Args:
     - mask_amount: we mask out 'mask_amount' of the total pixels (all RGB channels), default 92%.

    Returns:
     - mask: such that if a pixel is selected to be masked out, all 3 of its channels are zeroed out
    """
    mask = torch.rand(image.shape[0], 1, image.shape[2], image.shape[3], device=image.device) > mask_amount
    return mask

def random_type_mask(image, mask):
    """
    Applies the given random type inpainting mask to image. 
    """
    return image * mask

if __name__ == "__main__":
    # # Just generate one image with standard DPS
    dps = DPS()
    data_iter = iter(FFHQ_dataloader)
    y = next(data_iter)
    # start_h, start_w = box_type(y)
    # inpaint_folder = f"{start_h}_{start_w}"
    # os.mkdir(inpaint_folder)
    # box = box_type_mask(y, start_h, start_w).to('cuda:0')  
    # image_tools.save_image(box, f"{inpaint_folder}/y.jpg")
    # x = dps.sample_conditional_posterior(box, box_type_mask, {'start_h': start_h, 'start_w': start_w})
    # image_tools.save_image(x, f"{inpaint_folder}/inpainting_{s}.jpg", plot=True)

    # # Bounding Box Constant, Vary Factor
    # start_h = 41
    # start_w = 84
    # inpaint_folder = f"{start_h}_{start_w}"
    # os.mkdir(inpaint_folder)
    # box = box_type_mask(y, start_h, start_w).to('cuda:0')  
    # image_tools.save_image(box, f"{inpaint_folder}/y.jpg")
    # step_size = [0, 0.001, 0.01, 0.1, 1.0, 10.0, 50.0]
    # for s in step_size:
    #     dps_obj = DPS(step_size_factor=s)
    #     x = dps_obj.sample_conditional_posterior(box, box_type_mask, {'start_h': start_h, 'start_w': start_w})
    #     image_tools.save_image(x, f"{inpaint_folder}/inpainting_{s}.jpg", plot=True)

    # # Factor Constant, Vary Image and Box
    data_iter = iter(FFHQ_dataloader)
    for i in range(20):
        y = next(data_iter)
        if i < 8:
            continue
        start_h, start_w = box_type(y)
        inpaint_folder = f"{start_h}_{start_w}"
        # os.mkdir("vary_image")
        box = box_type_mask(y, start_h, start_w).to('cuda:0')  
        image_tools.save_image(box, f"vary_image/y{i}_{start_h}_{start_w}.jpg")
        x = dps.sample_conditional_posterior(box, box_type_mask, {'start_h': start_h, 'start_w': start_w})
        image_tools.save_image(x, f"vary_image/inpainting_{i}.jpg", plot=True)

    # # Generate for every image in dataset
    # for image in FFHQ_dataloader
    
import numpy as np
from dps import DPS
from image_loader import FFHQ_dataloader
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

def box_type(image, box_size=128, margin=16):
    """Box-type inpainting, as specified by DPS paper [1] and Chung et all 2022a [2], this means that:
     - we mask out a 128×128 box region from the image
     - the location of the box is sampled uniformly within 16 pixel margin of each side
    """
    h, w = image.shape[-2:]
    
    start_h = np.random.randint(margin, h - margin - box_size)
    start_w = np.random.randint(margin, w - margin - box_size)
    
    mask = torch.ones_like(image)
    mask[:, :, start_h:start_h+box_size, start_w:start_w+box_size] = 0
    return image * mask

def random_type(image, mask_amount = 0.92):
    mask = torch.rand(image.shape[0], 1, image.shape[2], image.shape[3], device=image.device) > mask_amount
    return image * mask

if __name__ == "__main__":
    dps = DPS()
    idx = 0
    # for image in FFHQ_dataloader
    data_iter = iter(FFHQ_dataloader)
    y = next(data_iter)

    # # Display the image
    # plt.imshow(np.transpose(y[0], (1, 2, 0)))
    # plt.axis('off')  # Turn off axis
    # plt.show()

    box = box_type(y)    
    plt.imshow(np.transpose(box[0], (1, 2, 0)))
    plt.axis('off')
    plt.show()

    # bruh = random_type(y)
    # plt.imshow(np.transpose(bruh[0], (1, 2, 0)))
    # plt.axis('off')

    x = dps.sample_conditional_posterior(box, box_type)
    plt.imshow(x)
    plt.axis('off')  # Turn off axis
    plt.show()


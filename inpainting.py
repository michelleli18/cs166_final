import numpy as np
from dps import DPS
from image_loader import FFHQ_dataloader
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

if __name__ == "__main__":
    dps = DPS()
    idx = 0
    # for image in FFHQ_dataloader
    data_iter = iter(FFHQ_dataloader)
    y = next(data_iter)

    # # Display the image
    # plt.imshow(y)
    # plt.axis('off')  # Turn off axis
    # plt.show()
    
    x = dps.sample_conditional_posterior(y)
    # plt.imshow(x)
    # plt.axis('off')  # Turn off axis
    # plt.show()


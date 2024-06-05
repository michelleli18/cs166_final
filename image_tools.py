from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def save_image(x, save_path, plot=False):
    x = x.cpu().detach().numpy()
    x = np.transpose(x[0], (1, 2, 0))
    print(x.shape)
    Image.fromarray(np.uint8(255 * x)).save(save_path)
    # .clip(0, 1)**(1/2.2)
    if plot == True:
        plt.imshow(x)
        plt.axis('off')  # Turn off axis
        plt.show()

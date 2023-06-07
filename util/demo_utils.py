"""
3DCoMPaT 2D Dataloader Demo
"""
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision.transforms as T


ROOT_URL = './mock_data/'


def make_transform(mode):
    if mode == "train":
        return T.Compose([])
    elif mode == "valid":
        return T.Compose([])


def show_tensors(ims, size):
    """
    Displaying an image in the notebook.
    """
    n_ims = len(ims)
    f, axarr = plt.subplots(1, n_ims)
    f.set_size_inches(size, size*n_ims)
    if n_ims == 1: axarr = [axarr]
    for im, ax in zip(ims, axarr):
        ax.axis('off')
        im = np.array(torch.squeeze(im).permute(1, 2, 0))
        im = im/np.amax(im)
        ax.imshow(im)


def mask_to_colors(mask):
    """
    Given a mask tensor of shape B, H, W,
    where B is the match dimension and HxW is the spatial dimension,
    map each integer value in the tensor to a color using the viridis colormap.
    """
    colormap = cm.get_cmap('viridis')

    # Get all unique values in the mask tensor
    unique_values = torch.unique(mask)
    for i, value in enumerate(unique_values):
        mask[mask == value] = i

    # Normalize the mask values to the range [0, 1]
    normalized_mask = mask.float() / mask.max()
    mask_array = normalized_mask.cpu().numpy()

    # Apply the colormap to the mask array
    colored_mask = colormap(mask_array)
    colored_mask_tensor = torch.from_numpy(colored_mask.transpose(0, 3, 1, 2))
    
    return colored_mask_tensor.unsqueeze(0)

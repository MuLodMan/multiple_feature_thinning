import torch
import torchvision.transforms.functional as F
opt_device = torch.device('cpu')
if torch.backends.mps.is_available():
    opt_device = torch.device("mps")
else:
    print ("MPS device not found.")

def binary_float_tensor_to_img(img:torch.Tensor):
    return F.to_pil_image(img*255,mode="F")

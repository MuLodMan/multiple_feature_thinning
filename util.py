import torch
import torchvision.transforms.functional as F
opt_device = torch.device('cpu')
if torch.backends.mps.is_available():
    opt_device = torch.device("mps")
else:
    print ("MPS device not found.")

def binary_float_tensor_to_img(img:torch.Tensor):
    (b,c,h,w) = img.shape
    return F.to_pil_image(img.view(c,h,w)*255,mode="F")
sigmoid_trend = 20

#for debug
scan_count = 3
import torch 
from torchvision import transforms
import os
import util
from PIL import Image
os.chdir('/Users/allenmu/workspace/thinning/pythonHandler')

opt_device = torch.device("cpu")

blured_kernel = torch.tensor(data=[[[[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]]],dtype=torch.float32,requires_grad=False)


convert_to_tensor = transforms.PILToTensor()

train_data_path = "fontWithUnicode/geetype/sample_character"

target_img_path = "fontWithUnicode/geetype/stroke_img"

def image_to_binaryTensor(fileName_path): 
    with Image.open(fileName_path,mode='r') as im:
       gray = im.convert('L')

       img_tensor:torch.Tensor = (convert_to_tensor(gray)).float()
 
       (c,h,w) = img_tensor.shape  

       blurred_tensor = torch.nn.functional.conv2d(img_tensor.view(1,c,h,w),blured_kernel).squeeze()

       
    return (torch.gt(input=blurred_tensor,other=120)).float().view(1,c,h,w)  # normalizing to binary matrix

if __name__ =='__main__' :
   img_file_names = [filename for filename in os.listdir(target_img_path) if filename.endswith('.png')]
   for img_name in img_file_names:
       train_tensor = image_to_binaryTensor(os.path.join(train_data_path,img_name))  #training Data

       target_tensor = image_to_binaryTensor(os.path.join(target_img_path,img_name)) #target Data


       for epoch in range(100):
           
           
           
#for debug
    #    train_img = util.binary_float_tensor_to_img(train_tensor)
    #    target_img = util.binary_float_tensor_to_img(target_tensor)
    #    train_img.show()
#for debug         
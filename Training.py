import torch 
from torchvision import transforms
import os
import util
from PIL import Image
import torch.optim as optim
import pre_feature_conv

os.chdir('/Users/allenmu/workspace/thinning/pythonHandler')

#to set mps environment fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
#to set mps environment fallback

blured_kernel = torch.tensor(data=[[[[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]]]],dtype=torch.float32,requires_grad=False)


convert_to_tensor = transforms.PILToTensor()

train_data_path = "fontWithUnicode/geetype/sample_character"

target_img_path = "fontWithUnicode/geetype/stroke_img"

weight_path = "weight/"

outputs_img_path = "output_images"

def image_to_binaryTensor(fileName_path): 
    with Image.open(fileName_path,mode='r') as im:
       gray = im.convert('L')

       img_tensor:torch.Tensor = (convert_to_tensor(gray)).float()
 
       (c,h,w) = img_tensor.shape  

       blurred_tensor = torch.nn.functional.conv2d(img_tensor.view(1,c,h,w),blured_kernel,padding='same').squeeze()

       
    return (torch.gt(input=blurred_tensor,other=120)).float().view(1,c,h,w).to(device=util.opt_device)  # normalizing to binary matrix

if __name__ =='__main__' :
   img_file_names = [filename for filename in os.listdir(target_img_path) if filename.endswith('.png')]
   feature_thinning_module = pre_feature_conv.thinning_conv().to(device=util.opt_device)

   criterion = torch.nn.MSELoss(reduction='sum')
   
   optimizer = optim.SGD(feature_thinning_module.parameters(),lr = 1/128,momentum=0.8)
   for img_name in img_file_names:
       train_tensor = image_to_binaryTensor(os.path.join(train_data_path,img_name))  #training Data

       target_tensor = image_to_binaryTensor(os.path.join(target_img_path,img_name)) #target Data
       
       for epoch in range(100):
         if(util.scan_count <=0 ):break
    
         outputs = feature_thinning_module(train_tensor)

         loss = criterion(outputs,target_tensor)

         optimizer.zero_grad()

         loss.backward()

         optimizer.step()

         if(epoch % 50 == 0):
            print(f'{epoch} loss is :{loss.item()} ')
       util.scan_count -= 1
       util.binary_float_tensor_to_img(outputs).save(f'{outputs_img_path}/{img_name}')
   
   torch.save(feature_thinning_module.state_dict(),weight_path+'iter_6.pth')

           
#for debug
    #    train_img = util.binary_float_tensor_to_img(train_tensor)
    #    target_img = util.binary_float_tensor_to_img(target_tensor)
    #    train_img.show()
#for debug         
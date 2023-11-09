import torch
import torch.nn as nn
import util
from scipy.ndimage import gaussian_filter



# foreground_weight_kernel = torch.tensor(data=[[[[2,10,2],[10,50,10],[2,10,2]]]],dtype=torch.float32)

class layer_conv(nn.Module):
   def __init__(self):
    self.peel_squence = peel_net_conv()

class peel_net_conv(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.edge_3_3_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding='same',bias=None)
      self.edge_5_5_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding='same',bias=None)
      self.diag_3_3_edge_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding='same',bias=None)
      self.diag_3_3_edge_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding='same',bias=None)
      self.__init_weight()

    def __init_weight(self):
       pass
    
    def forward(self,img:torch.Tensor):       
      (main_layer,diag_layer,ligth_layer) = self.get_feature_map(img.detach())
      
        
      return main_layer
    
    def get_feature_map(self,img:torch.Tensor):
      if ((img.gt(1.0)).int()).sum() > 0:
         raise ValueError("The tensor isn't binary tensor") 
      weight_layer = nn.functional.conv2d(input=img,
                                          weight=torch.tensor(data=[[[[1/2,5/2,1/2],[5/2,25/2,5/2],[1/2,5/2,1/2]]]],dtype=torch.float32,requires_grad=False),
                                          stride=1,
                                          padding='same')

      # main_map:>=49/2 diag_map:>=45/2 light_edge:remain
      # main_map = (weight_layer.gt(24.0)).float()

      diag_main_map = (weight_layer.gt(22.0)).float()

      # light_map =  torch.sub(input=img,other=diag_main_map,alpha=1)

      fore_ground_map = (weight_layer.gt(15.0)).float()


      #return (main_map,diag_map,light_map)
      
      #Four directions
      template_right_up_layer = nn.functional.conv2d(input=img,
                                                     weight=torch.tensor(data=[[[[1,1,2],[-1,-1,1],[-1,-1,1]]]],dtype=torch.float32,requires_grad=False),
                                                     stride=1,
                                                     padding='same')
      template_left_up_layer = nn.functional.conv2d(input=img,
                                                    weight=torch.tensor(data=[[[[2,1,1],[1,-1,-1],[1,-1,-1]]]],dtype=torch.float32,requires_grad=False),
                                                    stride=1,
                                                    padding='same')
      template_right_bottom_layer = nn.functional.conv2d(input=img,
                                                         weight=torch.tensor(data=[[[[-1,-1,1],[-1,-1,1],[1,1,2]]]],dtype=torch.float32,requires_grad=False),
                                                         stride=1,
                                                         padding='same')
      template_left_bottom_layer = nn.functional.conv2d(input=img,
                                                        weight=torch.tensor(data=[[[[1,-1,-1],[1,-1,-1],[2,1,1]]]],dtype=torch.float32,requires_grad=False),
                                                        stride=1,
                                                        padding='same'
                                                        )
      
      return  (fore_ground_map,
              (template_right_up_layer.gt(4)).float(),
              (template_left_up_layer.gt(4)).float(),
              (template_right_bottom_layer.gt(4)).float(),
              (template_left_bottom_layer.gt(4)).float())

      



  
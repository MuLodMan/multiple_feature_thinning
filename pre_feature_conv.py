import torch
import torch.nn as nn
import util
from scipy.ndimage import gaussian_filter


# foreground_weight_kernel = torch.tensor(data=[[[[2,10,2],[10,50,10],[2,10,2]]]],dtype=torch.float32)

class thinning_conv(nn.Module):
   def __init__(self):
    super().__init__()
    peel_list = [peel_net_conv() for i in range(3)]
    self.peel_squence_3 = nn.Sequential(*peel_list)

   def forward(self,img):
      return self.peel_squence_3(img) 

class peel_net_conv(torch.nn.Module):
    def __init__(self):
      super().__init__()

      self.up_right_5_5_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding='same',bias=None)
      self.up_right_7_7_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding='same',bias=None)

      self.up_left_5_5_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding='same',bias=None)
      self.up_left_7_7_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=7,stride=1,padding='same',bias=None)



      self.bottom_right_5_5_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding='same',bias=None)
      self.bottom_right_7_7_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding='same',bias=None)


      self.bottom_left_5_5_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=5,stride=1,padding='same',bias=None)
      self.bottom_left_7_7_kernel = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=7,stride=1,padding='same',bias=None)

      self.active = nn.Sigmoid()
  
      self.__init_weight()

    def __init_weight(self):
       pass
    
    def forward(self,img:torch.Tensor):       
      (t_r_u_map,t_l_u_map,t_r_b_map,t_l_b_map,thinned_ground_map) = self.get_feature_map(img.detach())
      bottom_left_layer = bottom_right_layer = up_left_layer = up_right_layer  = None
      # if(img.shape[0] == thinned_ground_map.shape[0]
      #    and img.shape[1] == thinned_ground_map.shape[1]
      #    and img.shape[2] == thinned_ground_map.shape[2]
      #    and img.shape[3] == thinned_ground_map.shape[3]):
      #   fore_ground_layer = img * thinned_ground_map
      # else:raise ValueError("the foreground error shape ")

      if(img.shape[0] == t_r_u_map.shape[0]
         and img.shape[1] == t_r_u_map.shape[1]
         and img.shape[2] == t_r_u_map.shape[2]
         and img.shape[3] == t_r_u_map.shape[3]):
        up_right_layer = self.up_right_7_7_kernel(
            self.up_right_5_5_kernel(img)) * t_r_u_map
      else:raise ValueError("the t_r_u map error shape ")
        
      if(img.shape[0] == t_l_u_map.shape[0]
         and img.shape[1] == t_l_u_map.shape[1]
         and img.shape[2] == t_l_u_map.shape[2]
         and img.shape[3] == t_l_u_map.shape[3]):
         up_left_layer = self.up_right_7_7_kernel(
            self.up_right_5_5_kernel(img)) * t_l_u_map
      else:raise ValueError("the t_l_u map error shape ")

      if(img.shape[0] == t_r_b_map.shape[0]
         and img.shape[3] == t_r_b_map.shape[3]
         and img.shape[2] == t_r_b_map.shape[2]
         and img.shape[1] == t_r_b_map.shape[1]):
         bottom_right_layer = self.up_right_7_7_kernel(
            self.up_right_5_5_kernel(img)) * t_r_b_map
      else:raise ValueError("the t_r_b map error shape ")

      if(img.shape[0] == t_l_b_map.shape[0]
         and img.shape[3] == t_l_b_map.shape[3]
         and img.shape[2] == t_l_b_map.shape[2]
         and img.shape[1] == t_l_b_map.shape[1]):
         bottom_left_layer = self.up_right_7_7_kernel(
            self.up_right_5_5_kernel(img)) * t_l_b_map
      else:raise ValueError("the t_l_b map error shape ")
      
        
      return self.active((img * thinned_ground_map - 
                         up_right_layer - up_left_layer - 
                         bottom_right_layer - bottom_left_layer-1/2)*util.sigmoid_trend)
    
    def get_feature_map(self,img:torch.Tensor):
      normalize_img = (img.gt(0.8)).float()
      weight_layer = nn.functional.conv2d(input=normalize_img,
                                          weight=torch.tensor(data=[[[[1/2,5/2,1/2],[5/2,25/2,5/2],[1/2,5/2,1/2]]]],
                                                              dtype=torch.float32,
                                                              requires_grad=False,
                                                              device=util.opt_device),
                                          stride=1,
                                          padding='same')

      # main_map:>=49/2 diag_map:>=45/2 light_edge:remain
      # main_map = (weight_layer.gt(24.0)).float()
      # diag_main_map = (weight_layer.gt(22.0)).float()
      # light_map =  torch.sub(input=img,other=diag_main_map,alpha=1)
      thinned_ground_map = (weight_layer.gt(22.0)).float()
      #return (main_map,diag_map,light_map)



      
      #Four directions
      template_right_up_layer = (nn.functional.conv2d(input=normalize_img,
                                                     weight=torch.tensor(data=[[[[1,1,1],[-1,-1,1],[-1,-1,1]]]],
                                                                         dtype=torch.float32,
                                                                         requires_grad=False,
                                                                         device=util.opt_device),
                                                     stride=1,
                                                     padding='same')).gt(4).float()
      template_left_up_layer = nn.functional.conv2d(input=normalize_img,
                                                    weight=torch.tensor(data=[[[[1,1,1],[1,-1,-1],[1,-1,-1]]]],
                                                                        dtype=torch.float32,
                                                                        requires_grad=False,
                                                                        device=util.opt_device),
                                                    stride=1,
                                                    padding='same').gt(4).float()
      template_right_bottom_layer = nn.functional.conv2d(input=normalize_img,
                                                         weight=torch.tensor(data=[[[[-1,-1,1],[-1,-1,1],[1,1,1]]]],
                                                                             dtype=torch.float32,
                                                                             requires_grad=False,
                                                                             device=util.opt_device),
                                                         stride=1,
                                                         padding='same').gt(4).float()
      template_left_bottom_layer = nn.functional.conv2d(input=normalize_img,
                                                        weight=torch.tensor(data=[[[[1,-1,-1],[1,-1,-1],[1,1,1]]]],
                                                                            dtype=torch.float32,
                                                                            requires_grad=False,
                                                                            device=util.opt_device
                                                                            ),
                                                        stride=1,
                                                        padding='same'
                                                        ).gt(4).float()
      
      return  (
              template_right_up_layer.roll(shifts=(-1,1),dims=(2,3)),
              template_left_up_layer.roll(shifts=(-1,-1),dims=(2,3)),
              template_right_bottom_layer.roll(shifts=(1,1),dims=(2,3)),
              template_left_bottom_layer.roll(shifts=(1,-1),dims=(2,3)),
              thinned_ground_map
              )

      



  
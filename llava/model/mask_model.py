import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from llava.model.segment_anything.build_sam import build_sam_vit_l

from llava.model.segment_anything.predictor import SamPredictor
from llava.model.segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from llava.model.segment_anything.utils.transforms import ResizeLongestSide

class MaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam = build_sam_vit_l("/home/shawn/nvme/vl_research/peft_llama/data/sam_vit_l_0b3195.pth")
        self.point_grids =  build_all_layer_point_grids( 16, 0, 1)
        self.points_per_batch = 256 # 32 
        self.transform =  ResizeLongestSide(self.sam.image_encoder.img_size)
        self.device = None
        #self.predictor = SamPredictor(self.sam)
        
    def forward(self, images):
        #todo support batched forward
        #image tensor back to pil image 
        bs = images.shape[0]
        self.device = images.device
        #self.point_grids = self.point_grids.unsqueeze(0).repeat(bs, 1, 1, 1) 
        processed = []
        for image in images:
            tensor = image
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype = torch.float32, device=tensor.device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype = torch.float32, device=tensor.device)
            tensor *= std[:, None, None]
            tensor += mean[:, None, None]
            image = tensor.cpu().numpy().transpose(1, 2, 0)
            image = np.uint8(image * 255)
            processed.append(image)
      
        # print("processed shape", images.shape)  # 8, 224, 224, 3
        #print ("points shape", self.point_grids[0].shape) # 256, 2
        
        images_torch = []
        for image in processed:
            input_image = self.transform.apply_image(image)
            input_image_torch = torch.as_tensor(input_image, device=self.device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
            images_torch.append(input_image_torch)
            
        orig_size = image.shape[:2]
        
        points_scale = [[224, 224]] #np.array(cropped_im_size)[None, ::-1] 
        points_for_image = self.point_grids[0] * points_scale

        transformed_points = self.transform.apply_coords(points_for_image, orig_size)
        
        in_points = torch.as_tensor(transformed_points, device=self.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        
        #print("in_points shape", in_points.shape) # 256, 2
       # print("in_labels shape", in_labels.shape) # 256
        
        batched_input = []
        for i in range(bs):
            batched_input.append({
                'image': images_torch[i],
                'original_size': orig_size,
                'point_coords': in_points[:, None, :],
                'point_labels': in_labels[:, None],
            })
        
        out = self.sam(batched_input, multimask_output=True)
        # masks, iou_preds = self.predictor.predict_torch(
        #     in_points[:, None, :],
        #     in_labels[:, None],
        #     multimask_output=True,
        #     return_logits=True,
        # )
        #print("out shape", out.shape)
        # point_queries.append(masks)
        # point_queries = torch.cat(point_queries, dim=0)    # 256,1, 256 

        return out #point_queries.squeeze(1)
   
   
   
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
 
class MaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam = build_sam_vit_l("/home/shawn/nvme/vl_research/peft_llama/data/sam_vit_l_0b3195.pth")
        self.point_grids =  build_all_layer_point_grids( 16, 0, 1)
        self.points_per_batch = 256 # 32 
        self.predictor = SamPredictor(self.sam)
        
    def forward(self, image):
        #image tensor back to pil image 
        tensor = image
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype = torch.float32, device=tensor.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype = torch.float32, device=tensor.device)
        tensor *= std[:, None, None]
        tensor += mean[:, None, None]
        image = tensor.cpu().numpy().transpose(1, 2, 0)
        image = np.uint8(image * 255)
        
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, 0, 512/1500
        )
        
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            x0, y0, x1, y1 = crop_box # 0, 0, 224, 224
            cropped_im = image[y0:y1, x0:x1, :] # 224, 224, 3
            cropped_im_size = cropped_im.shape[:2]
            self.predictor.set_image(cropped_im)

            # Get points for this crop
            points_scale = np.array(cropped_im_size)[None, ::-1] # [[224, 224]]
            points_for_image = self.point_grids[layer_idx] * points_scale
            
            point_queries = [] 
            # Generate masks for this crop in batches
            for (points,) in batch_iterator(self.points_per_batch, points_for_image):
                #batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
                transformed_points = self.predictor.transform.apply_coords(points, cropped_im_size)
                in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
                in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
                masks, iou_preds = self.predictor.predict_torch(
                    in_points[:, None, :],
                    in_labels[:, None],
                    multimask_output=True,
                    return_logits=True,
                )
                point_queries.append(masks)
            point_queries = torch.cat(point_queries, dim=0)    # 256,1, 256 
            self.predictor.reset_image()

        return point_queries.squeeze(1)
   
   
   
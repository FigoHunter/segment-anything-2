import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from .utils import WORKSPACE

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = os.path.join(WORKSPACE,'checkpoints/sam2_hiera_large.pt')
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

predictor = SAM2ImagePredictor(sam2_model)

def load_image(path, include_points, exclude_points):
    image = Image.open(path)
    image = np.array(image.convert("RGB"))
    predictor.set_image(image)

    input_points = np.array([*include_points, *exclude_points])
    include_count = len(include_points)
    exclude_count = len(exclude_points)
    input_labels = np.array([1] * include_count + [0] * exclude_count)


    masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,
    )
    mask = masks[0]
    print(np.max(mask))
    return mask
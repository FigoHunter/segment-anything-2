import os
import torch
from enum import Enum
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from .utils import WORKSPACE, createTempFolder

class Sam2ImageHandle:
    def __init__(self, ckpt, model_cfg, device="cuda"):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
        model = build_sam2(model_cfg, ckpt, device=device)

        self._model_cfg = model_cfg
        self._ckpt = ckpt
        self._device = device

        self._model = model
        self._predictor = SAM2ImagePredictor(model)

    @property
    def model(self):
        return self._model
    
    @property
    def predictor(self):
        return self._predictor
    
    @property
    def device(self):
        return self._device
    
    @property
    def model_cfg(self):
        return self._model_cfg
    
    @property
    def ckpt(self):
        return self._ckpt
    
    def predict(self, image_path, include_points, exclude_points, iteration=1):
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        self.predictor.set_image(image)
        input_points = np.array([*include_points, *exclude_points])
        include_count = len(include_points)
        exclude_count = len(exclude_points)
        input_labels = np.array([1] * include_count + [0] * exclude_count)
        mask_input=None
        for _ in range(iteration):
            masks, _, mask_input = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
                mask_input=mask_input,
            )
        mask = masks[0]
        return mask


def init_sam2_image():

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_checkpoint = os.path.join(WORKSPACE,'checkpoints/sam2_hiera_large.pt')
    model_cfg = "sam2_hiera_l.yaml"
    return Sam2ImageHandle(sam2_checkpoint, model_cfg)

def load_image(handle:Sam2ImageHandle, path, include_points, exclude_points, iteration=1):
    if len(include_points) == 0:
        return None
    input_points = np.array([*include_points, *exclude_points])
    mask = handle.predict(
        image_path=path,
        include_points=input_points,
        exclude_points=exclude_points,
        iteration=iteration,
    )
    return mask


class Sam2VideoHandle:
    def __init__(self, ckpt, model_cfg):
        from sam2.build_sam import build_sam2_video_predictor
        self._model_cfg = model_cfg
        self._ckpt = ckpt
        self._predictor = build_sam2_video_predictor(model_cfg, ckpt)
        self._infer_state = None
        self._video_path = None
        self._frame_names = None
        self._prompts = {}

    @property
    def model_cfg(self):
        return self._model_cfg
    
    @property
    def ckpt(self):
        return self._ckpt
    
    @property
    def predictor(self):
        return self._predictor
    
    @property
    def infer_state(self):
        return self._infer_state
    
    @property
    def video_path(self):
        return self._video_path


    def load_video(self, video_path):
        video_dir = createTempFolder('sam_video_rgb')
        os.system(f'ffmpeg -i {video_path} -q:v 2 -start_number 0 {video_dir}/%05d.jpg')
        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        self._prompts = {}

        self._video_path = video_path
        self._frame_names = frame_names
        self._infer_state = self._predictor.init_state(video_path=video_dir)
        self._predictor.reset_state(self._infer_state)

    def load_frames(self, frame_dir):
        frame_names = [
            p for p in os.listdir(frame_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        self._prompts = {}

        self._video_path = frame_dir
        self._frame_names = frame_names
        self._infer_state = self._predictor.init_state(video_path=frame_dir)
        self._predictor.reset_state(self._infer_state)



def init_sam2_video():

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    return Sam2VideoHandle(sam2_checkpoint, model_cfg)

def load_video(handle:Sam2VideoHandle, video_path):
    video_dir = createTempFolder('sam_video_rgb')
    os.system(f'ffmpeg -i {video_path} -q:v 2 -start_number 0 {video_dir}/%05d.jpg')

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    predictor = handle.predictor
    inference_state = predictor.init_state(video_path=video_dir)
    






# #### Step 1: Add a first click on a frame 
# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a positive click at (x, y) = (210, 350) to get started
# points = np.array([[210, 350]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# #### Step 2: Add a second click to refine the prediction
# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
# # sending all clicks (and their labels) to `add_new_points`
# points = np.array([[210, 350], [250, 220]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1, 1], np.int32)
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# #### Step 3: Propagate the prompts to get the masklet across the video
# # run propagation throughout the video and collect the results in a dict
# video_segments = {}  # video_segments contains the per-frame segmentation results
# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }

# #### Step 4: Add new prompts to further refine the masklet
# ann_frame_idx = 150  # further refine some details on this frame
# ann_obj_id = 1  # give a unique id to the object we interact with (it can be any integers)

# # Let's add a negative click on this frame at (x, y) = (82, 415) to refine the segment
# points = np.array([[82, 415]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([0], np.int32)
# _, _, out_mask_logits = predictor.add_new_points(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # run propagation throughout the video and collect the results in a dict
# video_segments = {}  # video_segments contains the per-frame segmentation results
# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }




# #### Step 1: Add two objects on a frame
# prompts = {}  # hold all the clicks we add for visualization

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a positive click at (x, y) = (200, 300) to get started on the first object
# points = np.array([[200, 300]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # add the first object
# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)

# # Let's add a 2nd negative click at (x, y) = (275, 175) to refine the first object
# # sending all clicks (and their labels) to `add_new_points`
# points = np.array([[200, 300], [275, 175]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1, 0], np.int32)
# prompts[ann_obj_id] = points, labels
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# ann_frame_idx = 0  # the frame index we interact with
# ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)

# # Let's now move on to the second object we want to track (giving it object id `3`)
# # with a positive click at (x, y) = (400, 150)
# points = np.array([[400, 150]], dtype=np.float32)
# # for labels, `1` means positive click and `0` means negative click
# labels = np.array([1], np.int32)
# prompts[ann_obj_id] = points, labels

# # `add_new_points` returns masks for all objects added so far on this interacted frame
# _, out_obj_ids, out_mask_logits = predictor.add_new_points(
#     inference_state=inference_state,
#     frame_idx=ann_frame_idx,
#     obj_id=ann_obj_id,
#     points=points,
#     labels=labels,
# )

# # run propagation throughout the video and collect the results in a dict
# video_segments = {}  # video_segments contains the per-frame segmentation results
# for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#     video_segments[out_frame_idx] = {
#         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#         for i, out_obj_id in enumerate(out_obj_ids)
#     }

import os
import torch
from enum import Enum
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from .utils import WORKSPACE, create_temp_folder

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
        self._obj_idx = None

    class Task:
        class State(Enum):
            NOT_STARTED = 0
            STARTED = 1

        def __init__(self, handle, frames):
            self._state = self.State.NOT_STARTED
            self._handle = handle
            self._frames = frames
            self._infer_state = None
            self._prompts = Sam2VideoPrompt(frame_count=len(frames), obj_idx=self.obj_idx)
            self._mask_cache = [None]*len(self.frames)
        
        
        @property
        def handle(self):
            return self._handle
        
        @property
        def frames(self):
            return self._frames
        
        @property
        def predictor(self):
            return self.handle.predictor
        
        @property
        def prompts(self):
            return self._prompts

        @property
        def state(self):
            return self._state
        
        @property
        def obj_idx(self):
            return self.handle.obj_idx

        def start(self):
            self._infer_state = self.predictor.init_state(image_paths=self._frames)
            print('infer_state:', self._infer_state)
            self.predictor.reset_state(self._infer_state)
            print('reset_state:', self._infer_state)
            self._state = self.State.STARTED

        def reset(self, *, clear_prompts=True, clear_cache=True):
            self.predictor.reset_state(self._infer_state)
            self._dirty = False
            if clear_prompts:
                self.prompts.clear()
            if clear_cache:
                self._mask_cache = [None]*len(self.frames)
            self._state = self.State.STARTED

        def update_selection(self):
            frame, obj, selection = self.prompts.get_selection()
            if selection is not None:
                points = np.array([*selection[0], *selection[1]], dtype=np.float32)
                include_count = len(selection[0])
                exclude_count = len(selection[1])
                labels = np.array([1] * include_count + [0] * exclude_count)
                frame, objs, masks = self.predictor.add_new_points(
                    inference_state=self._infer_state,
                    frame_idx=frame,
                    obj_id=obj,
                    points=points,
                    labels=labels,
                    clear_old_points=True,
                )
                masks = masks.cpu().numpy()
                self._cache_mask(frame, objs, masks)
                self._clean_cache_from(frame + 1)
            else:
                self._clean_cache_from(frame)
                self.predictor.add_new_points(
                    inference_state=self._infer_state,
                    frame_idx=frame,
                    obj_id=obj,
                    points=np.array([], dtype=np.float32),
                    labels=np.array([], np.int32),
                    clear_old_points=True,
                )
                objs, masks = self.get_current_masks()
                
            return objs, masks

        def get_current_frame(self):
            idx = self.prompts.selected_frame
            if idx < 0 or idx >= len(self.frames):
                raise ValueError(f"Invalid frame index {idx}")
            return self.frames[self.prompts.selected_frame]
    
        
        def get_current_frame_idx(self):
            return self.prompts.selected_frame
            
        def get_current_masks(self):
            frame = self.prompts.selected_frame
            objs, masks = self._get_cached_masks(frame)
            if objs is not None and masks is not None:
                return objs, masks
            print('current_frame:', frame)
            enumerate = self.predictor.propagate_in_video(
                self._infer_state, start_frame_idx=frame, max_frame_num_to_track=1)
            frame, obj_ids, masks = next(enumerate)
            print('processed_frame:', frame)
            masks = masks.cpu().numpy()
            self._cache_mask(frame, obj_ids, masks)
            return obj_ids, masks

        def propagate(self):
            frame = self.prompts.selected_frame
            print('current_frame:', frame)
            for frame, obj_ids, masks in self.predictor.propagate_in_video(
                self._infer_state, start_frame_idx=frame, max_frame_num_to_track=1):
                self.prompts.selected_frame += 1
                print('processed_frame:', frame)
                masks = masks.cpu().numpy()
                self._cache_mask(frame, obj_ids, masks)
                yield frame, obj_ids, masks


        # def get_mask(self, use_cache=True, process_if_needed=True):
        #     frame, obj, selection = self.prompts.get_selection()
        #     if use_cache:
        #         cached_mask = self._get_cached_mask(frame, obj)
        #         if cached_mask is not None:
        #             return cached_mask
        #     if not process_if_needed:
        #         return None
        #     if selection is not None:
        #         return None
        #     points = np.array([*selection[0], *selection[1]], dtype=np.float32)
        #     include_count = len(selection[0])
        #     exclude_count = len(selection[1])
        #     labels = np.array([1] * include_count + [0] * exclude_count)
        #     frame, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
        #         inference_state=self._infer_state,
        #         frame_idx=frame,
        #         obj_id=obj,
        #         points=points,
        #         labels=labels,
        #         clear_old_points=True,
        #     )
        #     self._cache_mask(frame, out_obj_ids, out_mask_logits)
        #     self._clean_cache_from(frame + 1)
        #     return out_mask_logits
        
        def next_frame(self):
            self.prompts.selected_frame += 1

        def prev_frame(self):
            self.prompts.selected_frame -= 1

        def next_obj(self):
            self.prompts.selected_obj += 1

        def prev_obj(self):
            self.prompts.selected_obj -= 1

        def _cache_mask(self, frame, objs, masks, *, clear_frame=True):
            if clear_frame | self._mask_cache[frame] is None:
                self._mask_cache[frame] = {}
            for obj, mask in zip(objs, masks):
                self._mask_cache[frame][obj] = mask

        def _clean_cache_from(self, frame):
            for i in range(frame, len(self.frames)):
                self._mask_cache[i] = None

        def _get_cached_mask(self, frame, obj):
            if self._mask_cache[frame] is None:
                return None
            return self._mask_cache[frame].get(obj, None)
        
        def _get_cached_masks(self, frame):
            objs = []
            masks = []
            if self._mask_cache[frame] is None:
                return None, None
            for obj, mask in self._mask_cache[frame].items():
                objs.append(obj)
                masks.append(mask)
            return objs, masks


        def __str__(self):
            return f"Task(frames={self.frames}, infer_state={self._infer_state})"


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

    @property
    def obj_idx(self):
        return self._obj_idx

    def parse_video(self, video_path):
        import glob
        video_dir = create_temp_folder('sam_video_rgb')
        os.system(f'ffmpeg -i {video_path} -q:v 2 -start_number 0 {video_dir}/%05d.jpg')
        return video_dir

    def load_frames(self, frame_dir,obj_idx=None,*, chunk_size = 400):
        frame_names = [
            p for p in os.listdir(frame_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        frame_names = [os.path.join(frame_dir, p) for p in frame_names]
        self._video_path = frame_dir
        self._frame_names = frame_names
        print(chunk_size)
        print(len(frame_names))
        frame_slices = [frame_names[i:min(i+chunk_size,len(frame_names)-1)] for i in range(0, len(frame_names), chunk_size)]
        self._obj_idx = obj_idx
        for i in range(len(frame_slices)):
            task = self.Task(self, frame_slices[i])
            yield task


class Sam2VideoPrompt(dict):
    def __init__(self, *, frame_count=None, obj_idx=None):
        super().__init__()
        self._selected_frame = 0
        self._selected_obj = 0
        self._frame_count = frame_count
        self._obj_idx = obj_idx
        

    @property
    def frame_count(self):
        return self._frame_count
    
    @property
    def obj_idx(self):
        return self._obj_idx

    @property
    def selected_frame(self):
        if self._selected_frame < 0 or self._selected_frame >= self.frame_count:
            self._selected_frame = 0
        return self._selected_frame
    
    @selected_frame.setter
    def selected_frame(self, value):
        if value < 0 or value >= self.frame_count:
            raise ValueError(f"Invalid frame index {value}")
        self._selected_frame = value

    @property
    def selected_obj(self):
        return self._selected_obj
    
    @selected_obj.setter
    def selected_obj(self, value):
        self._selected_obj = value

    def clear(self):
        self._selected_frame = -1
        super().clear()

    def iter_prompt(self):
        for frame, framedata in self.items():
            for obj_id, selection in framedata.items():
                yield (frame, obj_id, selection[0], selection[1])
                
    def get_selection(self, *, frame=None, obj=None):
        if frame is None:
            frame = self.selected_frame
        if obj is None:
            obj = self.selected_obj
        selected = self.get(frame, None)
        if selected is None:
            return frame, obj, None
        selected = selected.get(obj, None)
        if selected is None:
            return frame, obj, None
        return frame, obj, selected
    
    def set_selection(self, selection, *, frame=None, obj=None):
        if frame is None:
            frame = self.selected_frame
        if obj is None:
            obj = self.selected_obj
        if selection is None:
            if frame in self:
                if obj in self[frame]:
                    del self[frame][obj]
                    if len(self[obj]) == 0:
                        del self[obj]
        else:
            if self.selected_frame not in self:
                self[self.selected_frame] = {}
            self[self.selected_frame][self.selected_obj] = selection
            return self.selected_frame, self.selected_obj, selection

    def get_objs(self, frame=None):
        if frame is None:
            frame = self.selected_frame
        return self.get(frame, {}).keys()
    
    

def init_sam2_video():
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    sam2_checkpoint = os.path.join(WORKSPACE,'checkpoints/sam2_hiera_large.pt')
    model_cfg = "sam2_hiera_l.yaml"
    return Sam2VideoHandle(sam2_checkpoint, model_cfg)


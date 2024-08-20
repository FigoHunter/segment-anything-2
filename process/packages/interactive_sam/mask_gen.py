import os
import torch
from enum import Enum
import numpy as np
import weakref
# import matplotlib.pyplot as plt
import traceback
from PIL import Image
from .history import History
from .utils import WORKSPACE, create_temp_folder
from ._base import *

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
            masks, scores, mask_input = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
                mask_input=mask_input,
            )

        sorted_ind = np.argsort(scores)[::-1]
        mask = masks[sorted_ind]
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
            NOT_INIT = 0
            INIT = 1

        def __init__(self, handle, frames):
            self._state = self.State.NOT_INIT
            self._handle = weakref.ref(handle)
            self._frames = frames
            self._infer_state = None
            self._prompts = Sam2VideoPrompt(frame_count=len(frames), obj_idx=self.obj_idx)
            # self._mask_cache = [None]*len(self.frames)
            self._cache_relation = {}
            self._prompts_history = History(self._prompts, 100)
            self._cache_saver = AsyncCacheSave(self.frames)
            self._cache_saver.start()
        

        @property
        def handle(self):
            return self._handle()
        
        @property
        def frames(self):
            return self._frames
        
        @property
        def predictor(self):
            return self.handle.predictor
        
        @property
        def state(self):
            return self._state
        
        @property
        def obj_idx(self):
            return self.handle.obj_idx

        def start(self):
            if self.state != self.State.NOT_INIT:
                self.reset(clear_prompts=True, clear_cache=True)
                print('Task already started. Executed Task.reset()')
                return
            self._infer_state = self.predictor.init_state(image_paths=self._frames)
            print('infer_state:', self._infer_state)
            self.predictor.reset_state(self._infer_state)
            print('reset_state:', self._infer_state)
            self._state = self.State.INIT

        def reset(self, *, clear_prompts=True, clear_cache=True):
            if self.state == self.State.NOT_INIT:
                raise Exception("Task not started. Call Task.start() first")
            self.predictor.reset_state(self._infer_state)
            self._dirty = False
            if clear_prompts:
                self._prompts.clear()
            if clear_cache:
                # self._mask_cache = [None]*len(self.frames)
                self._cache_relation = {}
                self._prompts_history.clear()
                self._cache_saver = AsyncCacheSave([frame.replace('rgb','mask') for frame in self.frames])
                self._cache_saver.start()
            self._state = self.State.INIT

        def clear_frame(self, frame = None, clear_track=True):
            if frame is None:
                frame = self._prompts.selected_frame
            self._prompts.clear_frame(frame)
            if clear_track:
                self._clean_cache_relation(frame)
                self.predictor.clear_track(self._infer_state, frame)
                
        def set_selection(self, selection):
            self._prompts.set_selection(selection)

        def update_selection(self):
            objs = self._prompts.get_objs()
            frame = self._prompts.selected_frame
            for obj in objs:
                _,_,selection = self._prompts.get_selection(frame=frame, obj=obj)
                if selection is None:
                    print('Warning: No selection found for object', obj, 'reverting to previous state')
                    continue
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
            masks = (masks > 0.0).float().cpu().numpy()
            h,w = masks.shape[-2:]
            masks = masks.reshape(-1,h,w)
            # print('min:', masks.min(), 'max:', masks.max())
            # masks = masks.reshape(h,w)
            # print('masks:', masks.shape)
            self._clean_cache_relation(frame)
            self._cache_mask(frame, objs, masks)
            return objs, masks

        def get_selection(self):
            _, _, selection = self._prompts.get_selection()
            return selection

        def get_current_frame(self):
            idx = self._prompts.selected_frame
            if idx < 0 or idx >= len(self.frames):
                raise ValueError(f"Invalid frame index {idx}")
            return self.frames[self._prompts.selected_frame]
    
        def get_current_obj(self):
            idx = self._prompts.selected_obj
            return idx
        
        def get_current_frame_idx(self):
            return self._prompts.selected_frame
            
        def get_current_masks(self):
            frame = self._prompts.selected_frame
            objs, masks = self._get_cached_masks(frame)
            return objs, masks

        def is_keyframe(self, frame=None):
            return len(self._prompts.get_objs(frame))>0

        def propagate(self, reverse=False):
            frame = self._prompts.selected_frame
            print('current_frame:', frame)
            keyframe = frame

            for frame, obj_ids, masks in self.predictor.propagate_in_video(
                self._infer_state, start_frame_idx=frame, reverse=reverse):
                print('processed_frame:', frame)
                self._prompts.selected_frame = frame
                masks = (masks > 0.0).float().cpu().numpy()
                h,w = masks.shape[-2:]
                masks = masks.reshape(-1,h,w)
                # print(obj_ids)
                # print(masks.shape)
                # print('min:', masks.min(), 'max:', masks.max())
                # print('before cache')
                # print(str([[dict.keys()] if dict else None for dict in self._mask_cache]))
                # print(self._cache_relation)
                self._cache_mask(frame, obj_ids, masks, clear_frame=False)         
                if self.is_keyframe(frame):
                    keyframe = frame
                else:
                    self._add_cache_relation(frame, keyframe)
                # print('after cache')
                # print(str([[dict.keys()] if dict else None for dict in self._mask_cache]))
                # print(self._cache_relation)
                # print('====================')    
                yield frame, obj_ids, masks


        def register_history(self):
            self._prompts_history.register(self._prompts)

        def redo(self):
            prompts = self._prompts_history.redo(False)
            if prompts is None:
                return
            self._prompts = prompts
        
        def undo(self):
            prompts = self._prompts_history.undo(False)
            if prompts is None:
                return
            self._prompts = prompts

        def fast_left(self):
            start_index = self._prompts.selected_frame
            for frame in range(start_index-1, -1, -1):
                if self.is_keyframe(frame):
                    self._prompts.selected_frame = frame
                    return True
            self._prompts.selected_frame = start_index
            return False

        def fast_right(self):
            start_index = self._prompts.selected_frame
            for frame in range(start_index+1, len(self.frames)):
                if self.is_keyframe(frame):
                    self._prompts.selected_frame = frame
                    return True
            self._prompts.selected_frame = start_index
            return False
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
            if self._prompts.selected_frame >= len(self.frames) - 1:
                return False
            self._prompts.selected_frame += 1
            return True

        def prev_frame(self):
            if self._prompts.selected_frame <= 0:
                return False
            self._prompts.selected_frame -= 1
            return True

        def next_obj(self):
            next_obj = self._prompts.next_obj
            if next_obj == self._prompts.selected_obj:
                return False
            self._prompts.selected_obj = next_obj

        def prev_obj(self):
            prev_obj = self._prompts.prev_obj
            if prev_obj == self._prompts.selected_obj:
                return False
            self._prompts.selected_obj = prev_obj

        def save_files(self):
            self._cache_saver.save_files()

        def _cache_mask(self, frame, objs, masks, *, clear_frame=True):
            if clear_frame:
                self._cache_saver.delete_cache(frame)
                if self.is_keyframe(frame):
                    self._clean_cache_relation(frame)
            self._cache_saver.update_cache(objs, masks, frame)
            # if clear_frame | (self._mask_cache[frame] is None):
            #     self._mask_cache[frame] = {}
            #     if self.is_keyframe(frame):
            #         self._clean_cache_relation(frame)
            # for obj, mask in zip(objs, masks):
            #     self._mask_cache[frame][obj] = mask

        def _add_cache_relation(self, cur_frame, ref_frame):
            if ref_frame not in self._cache_relation:
                self._cache_relation[ref_frame] = set()
            self._cache_relation[ref_frame].add(cur_frame)

        def _clean_cache_relation(self, frame):
            if frame not in self._cache_relation:
                return
            self._cache_saver.delete_cache(frame)
            # self._mask_cache[frame] = None
            for f in self._cache_relation[frame]:
                self._cache_saver.delete_cache(f)
                # self._mask_cache[f] = None
            del self._cache_relation[frame]

        def _iter_cache_relation(self, ref_frame):
            if ref_frame not in self._cache_relation:
                return
            for frame in self._cache_relation[ref_frame]:
                yield frame

        def _get_cached_mask(self, frame, obj):
            objs,masks = self._cache_saver.get_cache(frame)
            if obj not in objs:
                return None
            idx = objs.index(obj)
            return masks[idx]
            # if self._mask_cache[frame] is None:
            #     return None
            # return self._mask_cache[frame].get(obj, None)
        
        def _get_cached_masks(self, frame):
            return self._cache_saver.get_cache(frame)
            # objs = []
            # masks = []
            # if self._mask_cache[frame] is None:
            #     return objs, masks
            # for obj, mask in self._mask_cache[frame].items():
            #     objs.append(obj)
            #     masks.append(mask)
            # return objs, masks


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
        if len(frame_slices) == 0:
            obj_idx = None
        self._obj_idx = obj_idx
        for i in range(len(frame_slices)):
            task = self.Task(self, frame_slices[i])
            yield task


class Sam2VideoPrompt(dict):
    def __init__(self, *, frame_count=None, obj_idx=None):
        super().__init__()
        self._selected_frame = 0
        self._selected_obj_idx = 0
        self._frame_count = frame_count
        self._obj_idx = obj_idx
        if self._obj_idx is not None:
            self._selected_obj_idx = 0
        

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
        if self._obj_idx is not None:
            if self._selected_obj_idx < 0 or self._selected_obj_idx >= len(self._obj_idx):
                self._selected_obj_idx = 0
            return self._obj_idx[self._selected_obj_idx]
        return self._selected_obj_idx
    
    @selected_obj.setter
    def selected_obj(self, value):
        idx = self._get_obj_idx(value)
        self._selected_obj_idx = idx
        self._selected_obj_idx = idx

    @property
    def next_obj(self):
        if self._obj_idx is None:
            return self._selected_obj_idx + 1
        idx = self._selected_obj_idx + 1
        if idx >= len(self._obj_idx):
            idx = len(self._obj_idx) - 1
        return self._obj_idx[idx]

    @property
    def prev_obj(self):
        if self._obj_idx is None:
            return self._selected_obj_idx - 1
        idx = self._selected_obj_idx - 1
        if idx < 0:
            idx = 0
        return self._obj_idx[idx]

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
    
    def clear_frame(self, frame=None):
        if frame is None:
            frame = self.selected_frame
        if frame in self:
            del self[frame]

    def set_selection(self, selection, *, frame=None, obj=None):
        if frame is None:
            frame = self.selected_frame
        if obj is None:
            obj = self.selected_obj
        if selection is None:
            if frame in self:
                if obj in self[frame]:
                    del self[frame][obj]
                    if len(self[frame]) == 0:
                        del self[frame]
        else:
            if self.selected_frame not in self:
                self[self.selected_frame] = {}
            self[self.selected_frame][self.selected_obj] = selection
            return self.selected_frame, self.selected_obj, selection

    def get_objs(self, frame=None):
        if frame is None:
            frame = self.selected_frame
        return self.get(frame, {}).keys()
    
    def _get_obj_idx(self, obj):
        if self._obj_idx is None:
            return obj
        idx = self._obj_idx.index(obj)
        if idx < 0:
            raise ValueError(f"Invalid object id {obj}")
        return self._obj_idx[idx]
        

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


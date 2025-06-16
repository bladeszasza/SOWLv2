"""
Wrapper for SAM2 (Segment Anything Model 2) from Facebook Research.
Handles both single-image and video segmentation.
"""
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sowlv2.utils.pipeline_utils import CUDA, CPU
# Unused imports: os, tempfile. They were W0611.

_SAM_MODELS = {
    # model-id ➜ (checkpoint filename, cfg filename, video_cfg_rel_path)
    "facebook/sam2.1-hiera-tiny": (
        "sam2.1_hiera_tiny.pt", "sam2.1_hiera_t.yaml", "configs/sam2.1/sam2.1_hiera_t.yaml"
    ),
    "facebook/sam2.1-hiera-small": (
        "sam2.1_hiera_small.pt", "sam2.1_hiera_s.yaml", "configs/sam2.1/sam2.1_hiera_s.yaml"
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml", "configs/sam2.1/sam2.1_hiera_b+.yaml"
    ),
    "facebook/sam2.1-hiera-large": (
        "sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml", "configs/sam2.1/sam2.1_hiera_l.yaml"
    )
}

class SAM2Wrapper:
    """
    One wrapper handles:
    • single-image predictor  (SAM2ImagePredictor)
    • lazy-built video predictor (build_sam2_video_predictor)
    Both share weights & device.
    """
    def __init__(self, model_name="facebook/sam2.1-hiera-small", device=CPU):
        if model_name not in _SAM_MODELS:
            raise ValueError(f"Unsupported SAM-2 model: {model_name}")
        ckpt_name, _, vid_cfg_rel = _SAM_MODELS[model_name]

        self._ckpt_path = hf_hub_download(model_name, ckpt_name, repo_type="model")

        self.device = torch.device(device)
        self._img_pred = SAM2ImagePredictor.from_pretrained(model_name)
        # Ensure vid_cfg_rel is used as per SAM2's build_sam2_video_predictor expectations
        # It might need the full path or a path relative to some sam2 config directory.
        # vid_cfg_rel is a relative path within the sam2 library for 2.1
        self._vid_pred = build_sam2_video_predictor(
            vid_cfg_rel, self._ckpt_path, device=self.device)

        if self.device.type == CUDA: # Check device type correctly
            # Move SAM2 model to GPU if requested
            self._img_pred.model.to(self.device) # model is already on self.device for _vid_pred


    # ---------- single-image ----------
    def segment(self, pil_image, box_xyxy):
        """Return binary mask (H×W uint8) for one box in one image."""
        img = np.array(pil_image)
        self._img_pred.set_image(img)
        # Ensure box_xyxy is a numpy array as expected by SAM2ImagePredictor
        masks, _, _ = self._img_pred.predict(box=np.asarray(box_xyxy)[None], multimask_output=False)
        if masks is None or len(masks) == 0:
            return None
        # SAM2 returns boolean masks, convert to uint8
        return (masks[0]).astype(np.uint8)


    def init_state(self, frames_dir):
        """Return state for SAM 2, load all frames"""
        return self._vid_pred.init_state(frames_dir)

    def add_new_box(self, state, frame_idx, box, obj_idx):
        """Adds new boxes"""
        # Ensure box is a NumPy array of shape (1, 4) and dtype float32 for SAM2 video predictor
        box_np = np.array(box, dtype=np.float32).reshape(1, 4)
        return self._vid_pred.add_new_points_or_box(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_idx,
            box=box_np, # Pass the formatted numpy array
        )

    def propagate_in_video(self, state):
        """Propagate the same selection in the whole video sequence"""
        return self._vid_pred.propagate_in_video(state)

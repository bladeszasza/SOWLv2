import os, tempfile, torch, numpy as np
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

_SAM_MODELS = {
    # model-id ➜ (checkpoint filename, cfg filename)
    "facebook/sam2.1-hiera-tiny": ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t.yaml", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "facebook/sam2.1-hiera-small": ("sam2.1_hiera_small.pt", "sam2.1_hiera_s.yaml", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "facebook/sam2.1-hiera-base-plus":  ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "facebook/sam2.1-hiera-large": ("sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml", "configs/sam2.1/sam2.1_hiera_l.yaml")
}

class SAM2Wrapper:
    """
    One wrapper handles:
    • single-image predictor  (SAM2ImagePredictor)
    • lazy-built video predictor (build_sam2_video_predictor)
    Both share weights & device.
    """
    def __init__(self, model_name="facebook/sam2.1-hiera-small", device="cpu"):
        if model_name not in _SAM_MODELS:
            raise ValueError(f"Unsupported SAM-2 model: {model_name}")
        ckpt_name, cfg_rel, vid_cfg_rel = _SAM_MODELS[model_name]
        # Download checkpoint if necessary (≈200 MB once)
        self._ckpt_path = hf_hub_download(model_name, ckpt_name, repo_type="model")
        self._cfg_path  = hf_hub_download(model_name, cfg_rel, repo_type="model")

        self.device = torch.device(device)
        self._img_pred = SAM2ImagePredictor.from_pretrained(model_name)
        self._vid_pred = build_sam2_video_predictor(vid_cfg_rel, self._ckpt_path, device=self.device)
        if device == "cuda":
            # Move SAM2 model to GPU if requested
            self._img_pred.model.to(torch.device("cuda"))

    # ---------- single-image ----------
    def segment(self, pil_image, box_xyxy):
        """Return binary mask (H×W uint8) for one box in one image."""
        img = np.array(pil_image)
        self._img_pred.set_image(img)
        masks, _, _ = self._img_pred.predict(box=np.asarray(box_xyxy)[None], multimask_output=False)
        if masks is None or len(masks) == 0:
            return None
        return (masks[0] > 0.5).astype(np.uint8)

    def init_state(frames_dir):
        """Return state for SAM 2, load all frames"""
        return self._vid_pred.init_state(frames_dir)

    def add_new_box(state, boxes):
        """Adds new boxes"""
        return self._vid_pred.add_new_points_or_box(state, boxes=boxes)

    def propagate_in_video(state):
        """Propagate the same selection int the whole video sequence"""
        return self._vid_pred.propagate_in_video(state)
   

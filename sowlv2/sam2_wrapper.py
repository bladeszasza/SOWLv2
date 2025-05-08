import os, tempfile, torch, numpy as np
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

_SAM_MODELS = {
    # model-id ➜ (checkpoint filename, cfg filename)
    "facebook/sam2.1-hiera-tiny": ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_t.yaml"),
    "facebook/sam2.1-hiera-small": ("sam2.1_hiera_small.pt", "sam2.1_hiera_s.yaml"),
    "facebook/sam2.1-hiera-base-plus":  ("sam2.1_hiera_base_plus.pt", "sam2.1_hiera_b+.yaml"),
    "facebook/sam2.1-hiera-large": ("sam2.1_hiera_large.pt", "sam2.1_hiera_l.yaml")
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
        ckpt_name, cfg_rel = _SAM_MODELS[model_name]
        # Download checkpoint if necessary (≈200 MB once)
        ckpt_path = hf_hub_download(model_name, ckpt_name, repo_type="model")
        cfg_path  = hf_hub_download(model_name, cfg_rel, repo_type="model")

        self.device = torch.device(device)
        self._img_pred = SAM2ImagePredictor.from_pretrained(model_name)
        if device == "cuda":
            # Move SAM model to GPU if requested
            self.predictor.model.to(torch.device("cuda"))

        
        self._vid_pred = None  # lazy

    # ---------- single-image ----------
    def segment(self, pil_image, box_xyxy):
        """Return binary mask (H×W uint8) for one box in one image."""
        img = np.array(pil_image)
        self._img_pred.set_image(img)
        masks, _, _ = self._img_pred.predict(box=np.asarray(box_xyxy)[None], multimask_output=False)
        if masks is None or len(masks) == 0:
            return None
        return (masks[0] > 0.5).astype(np.uint8)

    # ---------- video ----------
    def video_predictor(self):
        """Lazily construct and cache a SAM-2 VideoPredictor."""
        if self._vid_pred is None:
            # Re-use same cfg & ckpt used for image predictor
            cfg_path = self._img_pred.model.cfg_file
            ckpt_path = self._img_pred.model.ckpt_path
            self._vid_pred = build_sam2_video_predictor(cfg_path, ckpt_path, device=self.device)
        return self._vid_pred

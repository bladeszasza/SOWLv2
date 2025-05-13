import os, tempfile, torch, numpy as np
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import logging # Added for debugging

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
        
        self._ckpt_path = hf_hub_download(model_name, ckpt_name, repo_type="model")
        self._cfg_path  = hf_hub_download(model_name, cfg_rel, repo_type="model")

        self.device = torch.device(device)

        # Image Predictor
        logging.info(f"Initializing SAM2ImagePredictor with {model_name}")
        self._img_pred = SAM2ImagePredictor.from_pretrained(model_name)
        if hasattr(self._img_pred, 'model') and isinstance(self._img_pred.model, torch.nn.Module):
            logging.info(f"Casting SAM2ImagePredictor's model ({type(self._img_pred.model).__name__}) to device {self.device} and dtype float32.")
            self._img_pred.model.to(device=self.device, dtype=torch.float32)
        else:
            logging.warning("SAM2ImagePredictor does not have a 'model' attribute or it's not a torch.nn.Module.")

        # Video Predictor
        # build_sam2_video_predictor loads the model and moves it to device.
        # The returned object self._vid_pred is the SAM2VideoPredictor instance.
        # self._vid_pred.model is SamAutomaticMaskGeneratorVideo (or similar)
        # self._vid_pred.model.model is the actual Sam (e.g. SamHiera) instance.
        logging.info(f"Building SAM2VideoPredictor with config {vid_cfg_rel} and checkpoint {self._ckpt_path} for device {self.device}.")
        self._vid_pred = build_sam2_video_predictor(vid_cfg_rel, self._ckpt_path, device=self.device)
        
        actual_video_nn_model = None
        # Path to the actual nn.Module: self._vid_pred (SAM2VideoPredictor) -> .model (SamAutomaticMaskGeneratorVideo) -> .model (Sam/SamHiera)
        if hasattr(self._vid_pred, 'model') and hasattr(self._vid_pred.model, 'model') and \
           isinstance(self._vid_pred.model.model, torch.nn.Module):
            actual_video_nn_model = self._vid_pred.model.model 
        elif hasattr(self._vid_pred, 'model') and isinstance(self._vid_pred.model, torch.nn.Module):
             # Fallback if the structure is flatter (e.g., self._vid_pred.model is the nn.Module)
            actual_video_nn_model = self._vid_pred.model
        
        if actual_video_nn_model is not None:
            logging.info(f"Casting SAM2 video core NN model ({type(actual_video_nn_model).__name__}) to dtype float32. (It should already be on device {self.device})")
            actual_video_nn_model.to(dtype=torch.float32)
            # Verification log (optional, can be noisy)
            # try:
            #     sample_param_name, sample_param = next(actual_video_nn_model.named_parameters())
            #     logging.info(f"SAM2 video NN model sample parameter '{sample_param_name}' dtype after cast: {sample_param.dtype}, device: {sample_param.device}")
            # except StopIteration:
            #     logging.warning("SAM2 video NN model has no parameters to check dtype.")
        else:
            logging.warning("Could not find the underlying torch.nn.Module for SAM2 video predictor to cast its dtype.")
        

    # ---------- single-image ----------
    def segment(self, pil_image, box_xyxy):
        """Return binary mask (H×W uint8) for one box in one image."""
        img = np.array(pil_image)
        self._img_pred.set_image(img)
        # Ensure inputs to predict are on the correct device and dtype if necessary, though SAM2ImagePredictor should handle it.
        masks, _, _ = self._img_pred.predict(box=np.asarray(box_xyxy)[None], multimask_output=False)
        if masks is None or len(masks) == 0:
            return None
        return (masks[0] > 0.5).astype(np.uint8)

    def init_state(self, frames_dir):
        """Return state for SAM 2, load all frames"""
        # The state might contain features. If these features are bfloat16 from image_encoder, this is where it happens.
        logging.info(f"Initializing SAM2 video state from frames in: {frames_dir}")
        state = self._vid_pred.init_state(frames_dir)
        # Potentially inspect state.frame_feats dtype here if issues persist
        # if hasattr(state, 'frame_feats') and state.frame_feats is not None:
        #     logging.info(f"Initial state.frame_feats dtype: {state.frame_feats.dtype}")
        return state

    def add_new_box(self, state, frame_idx, box, obj_idx):
        """Adds new boxes"""
        # The box is converted to float32 tensor inside SAM2
        return self._vid_pred.add_new_points_or_box( 
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_idx,
            box=box,)

    def propagate_in_video(self, state):
        """Propagate the same selection int the whole video sequence"""
        # This is where the error occurs.
        # If state.frame_feats or other features within state are bfloat16,
        # and the model's forward pass uses them with float32 weights, error occurs.
        return self._vid_pred.propagate_in_video(state)
import os, tempfile, torch, numpy as np
from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2_video_predictor 
from sam2.sam2_image_predictor import SAM2ImagePredictor
import logging

_SAM_MODELS = {
    "facebook/sam2.1-hiera-tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "facebook/sam2.1-hiera-small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "facebook/sam2.1-hiera-base-plus":  ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "facebook/sam2.1-hiera-large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml", "configs/sam2.1/sam2.1_hiera_l.yaml")
}

class SAM2Wrapper:
    """
    One wrapper handles:
    • single-image predictor  (SAM2ImagePredictor)
    • lazy-built video predictor (build_sam2_video_predictor)
    Both share weights & device.
    """
    def __init__(self, model_name="facebook/sam2.1-hiera-base-plus", device="cpu"): # Changed default model
        if model_name not in _SAM_MODELS:
            raise ValueError(f"Unsupported SAM-2 model: {model_name}")
        ckpt_name, img_cfg_rel, vid_cfg_rel = _SAM_MODELS[model_name] # img_cfg_rel for SAM2ImagePredictor if needed, vid_cfg_rel for video
        
        self._ckpt_path = hf_hub_download(model_name, ckpt_name, repo_type="model")
        # Note: SAM2ImagePredictor.from_pretrained handles its own config loading internally.
        # self._img_cfg_path  = hf_hub_download(model_name, img_cfg_rel, repo_type="model") # Not directly used if from_pretrained works
        # self._vid_cfg_path  = hf_hub_download(model_name, vid_cfg_rel, repo_type="model") # build_sam2_video_predictor expects relative path to config within sam2 package or HF path

        self.device = torch.device(device)

        # Image Predictor
        logging.info(f"Initializing SAM2ImagePredictor with {model_name}")
        self._img_pred = SAM2ImagePredictor.from_pretrained(model_name) # This handles its own model loading and config
        if hasattr(self._img_pred, 'model') and isinstance(self._img_pred.model, torch.nn.Module):
            logging.info(f"Casting SAM2ImagePredictor's model ({type(self._img_pred.model).__name__}) to device {self.device} and dtype float32.")
            self._img_pred.model.to(device=self.device, dtype=torch.float32)
        else:
            logging.warning("SAM2ImagePredictor does not have a 'model' attribute or it's not a torch.nn.Module.")

        # Video Predictor
        # build_sam2_video_predictor expects the config file path relative to the sam2 library's config dir,
        # or a Hugging Face path that resolves to such a config.
        # The vid_cfg_rel from _SAM_MODELS is already this kind of path.
        logging.info(f"Building SAM2VideoPredictor with config {vid_cfg_rel} and checkpoint {self._ckpt_path} for device {self.device}.")
        self._vid_pred = build_sam2_video_predictor(vid_cfg_rel, self._ckpt_path, device=self.device)
        logging.info(f"SAM2VideoPredictor instance (self._vid_pred) type: {type(self._vid_pred)}")

        # Attempt to cast the entire video predictor (which should be an nn.Module) to float32.
        # The build_sam2_video_predictor already moves it to the specified device.
        if isinstance(self._vid_pred, torch.nn.Module):
            logging.info(f"Casting SAM2VideoPredictor ({type(self._vid_pred).__name__}) itself to dtype float32.")
            self._vid_pred.to(dtype=torch.float32) # Already on self.device

            # Verification
            try:
                # Check a sample parameter's dtype
                param_names = [name for name, _ in self._vid_pred.named_parameters()]
                if param_names:
                    sample_param_name = param_names[0]
                    # Use get_parameter to be safe, though direct access should also work
                    sample_param = self._vid_pred.get_parameter(sample_param_name)
                    logging.info(f"SAM2VideoPredictor sample parameter '{sample_param_name}' AFTER float32 cast - dtype: {sample_param.dtype}, device: {sample_param.device}")
                    if sample_param.dtype != torch.float32:
                        logging.warning(f"!! SAM2VideoPredictor parameter '{sample_param_name}' is {sample_param.dtype} even after attempting float32 cast. This could lead to dtype issues.")
                else:
                    logging.info("SAM2VideoPredictor has no named parameters. This is unusual for a main model object.")
            except Exception as e:
                logging.warning(f"Could not verify parameter dtype for SAM2VideoPredictor after casting: {e}", exc_info=True)
        else:
            logging.error(f"self._vid_pred (type: {type(self._vid_pred)}) is NOT a torch.nn.Module. Cannot cast its dtype to float32. This is unexpected and will likely cause issues.")
        

    # ---------- single-image ----------
    def segment(self, pil_image, box_xyxy):
        """Return binary mask (H×W uint8) for one box in one image."""
        img = np.array(pil_image)
        self._img_pred.set_image(img) # This should handle image conversion to tensor internally
        masks, _, _ = self._img_pred.predict(box=np.asarray(box_xyxy)[None], multimask_output=False)
        if masks is None or len(masks) == 0:
            return None
        # masks are bool (NumPy) from SAM2ImagePredictor, convert to uint8
        return masks[0].astype(np.uint8)


    # ---------- video ----------
    def init_state(self, frames_dir):
        """Return state for SAM 2, load all frames and potentially cast features."""
        logging.info(f"Initializing SAM2 video state from frames in: {frames_dir}")
        
        # Run init_state. If an ambient bfloat16 autocast is active, features might become bfloat16 here.
        state = self._vid_pred.init_state(frames_dir) # state is typically an 'InferenceState' object

        # Explicitly cast known feature tensors in the state object to float32 if they are bfloat16.
        # Common attribute names for features are 'frame_feats', 'image_embeddings', 'interim_frame_feats'.
        # This depends on the internal structure of SAM2's InferenceState.
        
        feature_attributes_to_check = ['frame_feats', 'interim_frame_feats'] # Add more if known
        
        for attr_name in feature_attributes_to_check:
            if hasattr(state, attr_name):
                feature_tensor_or_list = getattr(state, attr_name)
                if feature_tensor_or_list is not None:
                    if isinstance(feature_tensor_or_list, torch.Tensor):
                        if feature_tensor_or_list.dtype == torch.bfloat16:
                            logging.info(f"Casting state.{attr_name} from {feature_tensor_or_list.dtype} to float32.")
                            setattr(state, attr_name, feature_tensor_or_list.to(torch.float32))
                        logging.info(f"State attribute state.{attr_name} - dtype: {getattr(state, attr_name).dtype}, shape: {getattr(state, attr_name).shape}, device: {getattr(state, attr_name).device}")
                    elif isinstance(feature_tensor_or_list, list):
                        new_list = []
                        changed_in_list = False
                        for i, item in enumerate(feature_tensor_or_list):
                            if isinstance(item, torch.Tensor) and item.dtype == torch.bfloat16:
                                logging.info(f"Casting element {i} of state.{attr_name} from {item.dtype} to float32.")
                                new_list.append(item.to(torch.float32))
                                changed_in_list = True
                            else:
                                new_list.append(item) # Keep as is
                            if isinstance(new_list[-1], torch.Tensor):
                                logging.info(f"State attribute state.{attr_name}[{i}] - dtype: {new_list[-1].dtype}, shape: {new_list[-1].shape}, device: {new_list[-1].device}")
                        if changed_in_list:
                            setattr(state, attr_name, new_list)
        return state

    def add_new_box(self, state, frame_idx, box, obj_idx):
        """Adds new boxes"""
        # Ensure box tensor is float32 if it's a tensor, SAM2 likely expects float coords
        if isinstance(box, torch.Tensor) and box.dtype != torch.float32:
            box_input = box.to(torch.float32)
        elif isinstance(box, np.ndarray) and box.dtype != np.float32:
            box_input = box.astype(np.float32)
        else:
            box_input = box # Assume list of floats or already correct type
            
        return self._vid_pred.add_new_points_or_box( 
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_idx,
            box=box_input) # Pass the potentially casted box

    def propagate_in_video(self, state):
        """Propagate the same selection in the whole video sequence, ensuring output dtype."""
        # The self._vid_pred.propagate_in_video is a generator
        for fidx, obj_ids, mask_logits in self._vid_pred.propagate_in_video(state):
            # Ensure mask_logits are float32, as they might become bfloat16 from internal SAM2 ops
            # if an ambient bfloat16 autocast was active or if SAM2 defaults to it internally.
            if isinstance(mask_logits, torch.Tensor) and mask_logits.dtype == torch.bfloat16:
                logging.debug(f"Casting mask_logits for frame {fidx} from {mask_logits.dtype} to float32.")
                mask_logits = mask_logits.to(torch.float32)
            yield fidx, obj_ids, mask_logits
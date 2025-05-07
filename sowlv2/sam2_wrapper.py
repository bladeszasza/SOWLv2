import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Wrapper:
    """Wrapper for SAM 2 segmentation."""
    def __init__(self, model_name="facebook/sam2.1-hiera-small", device="cpu"):
        self.device = device
        self.predictor = SAM2ImagePredictor.from_pretrained(model_name)
        if device == "cuda":
            # Move SAM model to GPU if requested
            self.predictor.model.to(torch.device("cuda"))

    def segment(self, image, box):
        """
        Generate a segmentation mask for the given box in the image.
        Returns a binary mask (numpy array) of shape (H, W).
        """
        # Ensure image is a numpy array (HxWRGB)
        if isinstance(image, np.ndarray):
            img_array = image
        else:
            img_array = np.array(image)
        # Set image in predictor
        self.predictor.set_image(img_array)
        # Prepare box in XYXY format
        box_input = np.array(box, dtype=float)
        masks, _, _ = self.predictor.predict(box=box_input, multimask_output=False)
        if masks is None or len(masks) == 0:
            return None
        mask = masks[0]  # First (and only) mask
        mask = (mask > 0.5).astype(np.uint8)  # Convert to 0/1
        return mask

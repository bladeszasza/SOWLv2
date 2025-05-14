"""
Wrapper for OWLv2 text-conditioned object detection models from HuggingFace Transformers.
"""
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class OWLV2Wrapper:  # pylint: disable=too-few-public-methods
    """Wrapper for OWLv2 text-conditioned object detection."""
    def __init__(self, model_name="google/owlv2-base-patch16-ensemble", device="cpu"):
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)

    def detect(self, image, prompt, threshold=0.4):
        """
        Detect objects in the image matching the text prompt.
        Returns a list of dict with keys: box, score, label.
        """
        text_labels = [[f"a photo of {prompt}"]]
        inputs = self.processor(
            text=text_labels, images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width)]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes,
            threshold=threshold, text_labels=text_labels
        )

        detections = []
        if results and results[0]:  # Check if results and the first element exist
            result = results[0]
            boxes = result["boxes"].cpu().numpy()  # Nx4 (xmin, ymin, xmax, ymax)
            scores = result["scores"].cpu().numpy()
            # Ensure labels are correctly accessed, assuming "text_labels" is directly in result items
            # or falls back to the input prompt if specific per-box labels aren't returned by post_process
            returned_labels = result.get("text_labels", [prompt] * len(boxes))

            for box, score, label_text in zip(boxes, scores, returned_labels):
                detections.append({
                    "box": [float(coord) for coord in box], # Ensure box coords are float
                    "score": float(score),
                    "label": label_text
                })
        return detections
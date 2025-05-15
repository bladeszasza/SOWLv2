"""
Wrapper for OWLv2 text-conditioned object detection models from HuggingFace Transformers.
"""
from typing import Union, List
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch

class OWLV2Wrapper:  # pylint: disable=too-few-public-methods
    """Wrapper for OWLv2 text-conditioned object detection."""
    def __init__(self, model_name="google/owlv2-base-patch16-ensemble", device="cpu"):
        self.device = device
        self.processor: Owlv2Processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)

    def detect(self, *, image, prompt: Union[str, List[str]], threshold=0.1):
        """
        Detect objects in the image matching the text prompt.
        Returns a list of dict with keys: box, score, label.
        """
        if isinstance(prompt, str):
            processed_prompts = [f"a photo of {prompt}"]
        else: # prompt is a list of strings
            processed_prompts = [f"a photo of {p}" for p in prompt]

        text_labels = [processed_prompts] # Batch size of 1, with potentially multiple queries

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

        # Determine the list of original prompt terms for fallback in _format_detections
        if isinstance(prompt, str):
            fallback_labels = [prompt]
        else:
            fallback_labels = prompt

        return self._format_detections(results, fallback_labels)

    def _format_detections(self, results, fallback_prompts: List[str]):
        """
        Helper to format detection results into a list of dicts.
        fallback_prompts: The list of original prompt terms used for searching.
        """
        detections = []
        if results and results[0]:
            result = results[0]
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            # 'text_labels' in result should be populated by post_process_object_detection
            # with the specific query that matched each box (e.g., "a photo of cat").
            # The processor.post_process_grounded_object_detection
            # returns the text_labels as they were passed in.
            returned_labels = result.get("text_labels", fallback_prompts * len(boxes)) # Fallback
            for box, score, label_text in zip(boxes, scores, returned_labels):
                detections.append({
                    "box": [float(coord) for coord in box],
                    "score": float(score),
                    "label": label_text 
                })
        return detections

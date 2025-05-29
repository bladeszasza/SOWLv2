"""
Wrapper for OWLv2 text-conditioned object detection models from HuggingFace Transformers.
"""
from typing import Union, List, Dict, Any
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch

# It's a focused wrapper, so R0903 (too-few-public-methods) might be flagged
# but is acceptable for this type of class. We can add the disable if Pylint complains.
# pylint: disable=R0903

class OWLV2Wrapper:
    """
    Wrapper for OWLv2 text-conditioned object detection.

    This class handles the loading of OWLv2 models and processors,
    and provides a method to detect objects based on text prompts.
    It formats the output to include both the full label matched by OWLv2
    and the original "core" prompt term provided by the user.
    """
    def __init__(self, model_name: str ="google/owlv2-base-patch16-ensemble", device: str = "cpu"):
        """
        Initialize the OWLV2Wrapper.

        Args:
            model_name (str): The Hugging Face model identifier for OWLv2.
            device (str): The device to run the model on (e.g., "cpu", "cuda").
        """
        self.device = device
        self.processor: Owlv2Processor = Owlv2Processor.from_pretrained(model_name)
        self.model: Owlv2ForObjectDetection = Owlv2ForObjectDetection.from_pretrained(
            model_name).to(
            self.device
        )

    def detect(self, *, image: Any, prompt: Union[str, List[str]], threshold: float = 0.1
               ) -> List[Dict[str, Any]]:
        """
        Detect objects in the image matching the text prompt(s).

        Args:
            image (Any): The input image (e.g., a PIL Image).
            prompt (Union[str, List[str]]): A single text prompt or a list of text prompts.
            threshold (float): The confidence threshold for detections.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            represents a detected object and contains 'box', 'score', 'label'
            (the full text matched by OWLv2), and 'core_prompt' (the original
            user-provided term that led to this detection).
        """
        if isinstance(prompt, str):
            original_prompt_terms: List[str] = [prompt]
        else:
            original_prompt_terms: List[str] = prompt

        # OWLv2 typically expects prompts like "a photo of <object>"
        processed_prompts_for_owl: List[str] = [
            f"a photo of {p}" for p in original_prompt_terms
        ]
        # The 'text' argument to the processor for multiple queries on a single image
        # should be List[List[str]], where the outer list is for batch items.
        text_labels_for_owl: List[List[str]] = [processed_prompts_for_owl]

        inputs = self.processor(
            text=text_labels_for_owl, images=image, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # target_sizes should be a tensor of shape (batch_size, 2)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)

        # Pass text_labels_for_owl to post_process for correct label association
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            text_labels=text_labels_for_owl
        )

        return self._format_detections(results, original_prompt_terms)

    def _format_detections(self, results: List[Dict[str, Any]],
                           original_prompt_terms: List[str]) -> List[Dict[str, Any]]:
        """
        Helper to format raw detection results into a structured list of dictionaries.

        Args:
            results (List[Dict[str, Any]]): Raw results from the OWLv2 processor's
                                            post_process_grounded_object_detection method.
            original_prompt_terms (List[str]): The list of original, user-provided
                                               prompt terms (e.g., ["cat", "dog"]).

        Returns:
            List[Dict[str, Any]]: Formatted list of detections.
        """
        detections: List[Dict[str, Any]] = []
        if not results or not results[0]:
            return detections

        # Results is a list (batch), we typically process one image at a time here.
        first_image_results = results[0]
        boxes = first_image_results["boxes"].cpu().numpy()
        scores = first_image_results["scores"].cpu().numpy()

        # 'labels' are integer indices into the list of queries *for the current image*
        # that were passed to post_process_grounded_object_detection
        # (i.e., text_labels_for_owl[0]).
        prompt_indices = first_image_results.get(
            "labels", torch.zeros(len(boxes), dtype=torch.long)
        ).cpu().numpy()

        # 'text_labels' from results should be the actual prompt strings that matched.
        owl_matched_full_labels = first_image_results.get("text_labels", [])

        for i, (current_box, current_score) in enumerate(zip(boxes, scores)):
            try:
                core_prompt = original_prompt_terms[prompt_indices[i]]
            except IndexError:
                core_prompt = "unknown_prompt_term"
                print(
                    f"Warning: Index {prompt_indices[i]} out of bounds"
                )

            full_owl_label = (
                owl_matched_full_labels[i]
                if i < len(owl_matched_full_labels)
                else f"a photo of {core_prompt}"
            )

            detections.append({
                "box": [float(coord) for coord in current_box],
                "score": float(current_score),
                "label": full_owl_label,
                "core_prompt": core_prompt
            })
        return detections

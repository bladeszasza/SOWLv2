import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

class OWLV2Wrapper:
    """Wrapper for OWLv2 text-conditioned object detection."""
    def __init__(self, model_name="google/owlv2-base-patch16-ensemble", device="cpu"):
        self.device = torch.device(device)
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device=self.device, dtype=torch.float32)

    def detect(self, image, prompt, threshold=0.4):
        """
        Detect objects in the image matching the text prompt.
        Returns a list of dict with keys: box, score, label.
        """
        text_labels = [[f"a photo of {prompt}"]]
        inputs = self.processor(text=text_labels, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([(image.height, image.width)]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold, text_labels=text_labels
        )
        detections = []
        if results:
            result = results[0]
            boxes = result["boxes"].cpu().numpy()  # Nx4 (xmin, ymin, xmax, ymax)
            scores = result["scores"].cpu().numpy()
            labels = result["text_labels"]
            for box, score, label in zip(boxes, scores, labels):
                detections.append({
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "score": float(score),
                    "label": label
                })
        return detections

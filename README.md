# SOWLv2

  <div>
    <a href="https://github.com/bladeszasza/SOWLv2/blob/main/notebooks/SOWLv2_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  </div>
  <br>

SOWLv2 (**S**egmented**OWLv2**) is a tool for **text-prompted object segmentation** that combines Google’s [OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2) (open-vocabulary object detector) with Meta’s [SAM 2](https://github.com/facebookresearch/sam2) (Segment Anything Model) to segment detected objects in images, frames, or videos. Given a text prompt (e.g. `"plant"`) and an input (image, folder of frames, or video), SOWLv2 will:

- Use OWLv2 to detect bounding boxes for objects matching the text prompt. Based on paper [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)
- Use SAM 2 to generate segmentation masks for each detected object. Based on paper [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
- Save both the **binary masks** (foreground vs background) and **overlay images** (original image with mask overlaid) to the output directory.

## Installation

SOWLv2 can be installed via pip from the Git repository:

```
pip install git+https://github.com/bladeszasza/SOWLv2
```

This will also install the required dependencies (transformers, sam2, opencv-python, etc.). Alternatively, install from the provided setup.py or use requirements.txt.
Usage

After installation, the command-line tool sowlv2-detect is available. Its basic usage is:

```
sowlv2-detect --prompt "text_prompt" --input <input_path> --output <output_dir> [options]
```

--prompt: The text query (string) for object detection (e.g. "cat", "tree", "red car").
--input: Path to input. This can be:
    - A single image file (e.g. image.jpg).
    - A directory of image frames (e.g. ./frames/).
    - A video file (e.g. video.mp4).
--output: Directory where outputs (masks and overlays) will be saved. The directory will be created if it does not exist.
--owl-model: (Optional) OWLv2 model name (HuggingFace). Default is google/owlv2-base-patch16-ensemble.
--sam-model: (Optional) SAM 2 model name (HuggingFace). Default is facebook/sam2.1-hiera-small.
--threshold: (Optional) Detection confidence threshold (0 to 1). Default is 0.1.
--fps: (Optional) Frame sampling rate for videos. Default is 24.
--device: (Optional) Compute device ("cuda" or "cpu"). Default uses GPU if available.

For example, to segment all "dogs" in an image:
```
sowlv2-detect --prompt "dog" --input dog.jpg --output results/
```

To run on a folder of frames:

```
sowlv2-detect --prompt "person" --input ./frames_folder/ --output results_frames/
```

To run on a video (sampling at 10 FPS):

```
sowlv2-detect --prompt "car" --input video.mp4 --output results_video/ --fps 10
```

The outputs will be saved in the specified output directory. For each detected object, you will find files like image_object0_mask.png (binary mask) and image_object0_overlay.png (overlay image). Objects are numbered in the order detected.

## Examples

Image Input: Detect and segment "cat" in cat.jpg. Outputs masks/overlays named cat_object0_mask.png, etc.
Frames Folder: Given frames/ containing frame001.jpg, frame002.jpg, ... output files like frame001_object0_overlay.png and frame001_object0_mask.png.
Video: Given video.mp4, frames are sampled at the specified --fps; outputs are named with the video base name and frame index.
See the Colab Notebook for a step-by-step demonstration on running SOWLv2 with all three input types.

## How It Works

OWLv2 Detection: We use the Hugging Face Owlv2Processor and Owlv2ForObjectDetection models to perform zero-shot object detection with text queries
huggingface.co
. The specified OWLv2 model (default google/owlv2-base-patch16-ensemble) processes the image and outputs bounding boxes for any objects matching the prompt text.
SAM 2 Segmentation: For each detected bounding box, we invoke the SAM 2 model to generate a segmentation mask. We use the sam2 Python package and SAM2ImagePredictor from Meta to load the pre-trained SAM model
huggingface.co
. We feed the image and each bounding box as a prompt to obtain a binary mask of the object.
Saving Results: Each mask is saved as a grayscale PNG (255 for object, 0 for background). An overlay image is also created by blending the mask with the original image (e.g., object region colored with transparency) and saved.

## Dependencies

SOWLv2 depends on:
torch (PyTorch)
transformers (for OWLv2)
sam2 (Meta’s SAM 2 package)
opencv-python (for video handling)
numpy, Pillow, pyyaml, huggingface_hub
These are specified in setup.py and requirements.txt.

## Acknowledgments

Our implementation is based on several awesome repositories:

- [OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2) 
- [SAM 2](https://github.com/facebookresearch/sam2)

We thank the respective authors for open-sourcing their code.

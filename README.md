<!-- # SOWLv2

TL;DR: SOWLv2: Text-prompted object segmentation using OWLv2 and SAM 2 -->

<p align="center">
  <h1 align="center">SOWLv2: Text-Prompted Object Segmentation from video</h2>
  <div>
    <a href="https://colab.research.google.com/drive/1vX6P4KNmWoisY-Vfq6bAVunsHaLrC-AO"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <!-- Add other badges if you have them, e.g., PyPI version, license, build status -->
    <!-- <a href="LINK_TO_PYPI"><img src="https://img.shields.io/pypi/v/sowlv2" alt="PyPI version"></a> -->
    <!-- <a href="LINK_TO_LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a> -->
  </div>
  <br>
</p>

SOWLv2 (**S**egmented**OWLv2**) is a powerful command-line tool for **text-prompted object segmentation**. It seamlessly integrates Google’s [OWLv2](https://huggingface.co/docs/transformers/en/model_doc/owlv2) open-vocabulary object detector with Meta’s [SAM 2](https://github.com/facebookresearch/sam2) (Segment Anything Model V2) to precisely segment objects in images, image sequences (frames), or videos based on natural language descriptions.

Given a text prompt (e.g., `"a red bicycle"`) and an input source, SOWLv2 will:
1.  Utilize **OWLv2** to detect bounding boxes for objects matching the text prompt, based on the principles from the paper [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683).
2.  Employ **SAM 2** to generate detailed segmentation masks for each detected object, leveraging techniques from the paper [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714).
3.  Save both **binary segmentation masks** (foreground vs. background) and **overlay images** (original image with masks visually overlaid) to a specified output directory.

## ✨ Key Features

*   **Text-Prompted Segmentation:** Identify and segment objects using free-form text descriptions.
*   **State-of-the-Art Models:** Leverages the power of Google's OWLv2 and Meta's SAM 2.
*   **Versatile Input:** Supports single images, directories of frames, and video files.
*   **Comprehensive Output:** Generates both binary masks for programmatic use and visual overlays for inspection.
*   **Customizable:** Allows selection of specific OWLv2 and SAM 2 model variants, detection thresholds, and video processing parameters.
*   **Easy Installation:** Installable via pip directly from the Git repository.
*   **GPU Acceleration:** Automatically utilizes CUDA-enabled GPUs if available, with a fallback to CPU.

## 🚀 Quick Start & Demo

Explore SOWLv2's capabilities interactively with our Google Colab Notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vX6P4KNmWoisY-Vfq6bAVunsHaLrC-AO)

The notebook provides a step-by-step demonstration for all supported input types (images, frames, videos).

## 🛠️ Installation

SOWLv2 can be installed directly from this Git repository using pip. Ensure you have Python 3.8+ and pip installed.

```bash
pip install git+https://github.com/bladeszasza/SOWLv2.git
```

This command will also install all necessary dependencies, including `transformers`, `sam2`, `opencv-python`, `torch`, and others.

Alternatively, you can clone the repository and install using `setup.py` or `requirements.txt`:
```bash
git clone https://github.com/bladeszasza/SOWLv2.git
cd SOWLv2
pip install -r requirements.txt
# or
# python setup.py install
```

## ⚙️ Usage

Once installed, the `sowlv2-detect` command-line tool will be available.

### Basic Command Structure:

```bash
sowlv2-detect --prompt "your text prompt" --input <path_to_input> --output <path_to_output_dir> [options]
```

### Command-Line Options:

| Argument        | Description                                                                                                | Default Value                        |
|-----------------|------------------------------------------------------------------------------------------------------------|--------------------------------------|
| `--prompt`      | **(Required)** Text query for object detection (e.g., `"cat"`, `"tree"`, `"red car"`).                       | `None`                               |
| `--input`       | **(Required)** Path to the input: a single image file, a directory of image frames, or a video file.        | `None`                               |
| `--output`      | Directory where outputs (masks and overlays) will be saved. Created if it doesn't exist.                   | `output/`                            |
| `--owl-model`   | (Optional) OWLv2 model name from Hugging Face Model Hub.                                                   | `google/owlv2-base-patch16-ensemble` |
| `--sam-model`   | (Optional) SAM 2 model name from Hugging Face Model Hub.                                                   | `facebook/sam2.1-hiera-small`        |
| `--threshold`   | (Optional) Detection confidence threshold for OWLv2 (a float between 0 and 1).                             | `0.1`                                |
| `--fps`         | (Optional) Frame sampling rate (frames per second) for video inputs.                                       | `24`                                 |
| `--device`      | (Optional) Compute device (`"cuda"` or `"cpu"`).                                                             | Auto-detects GPU, else `cpu`         |
| `--config`      | (Optional) Path to a YAML configuration file to specify arguments (see [Configuration](#configuration)).   | `None`                               |

### Examples:

1.  **Segment "dogs" in a single image:**
    ```bash
    sowlv2-detect --prompt "dog" --input path/to/your/dog_image.jpg --output results/dog_segmentation/
    ```

2.  **Segment "person" in a folder of image frames:**
    ```bash
    sowlv2-detect --prompt "person" --input path/to/your/frames_folder/ --output results/person_frames_segmentation/
    ```

3.  **Segment "car" in a video, sampling at 10 FPS:**
    ```bash
    sowlv2-detect --prompt "car" --input path/to/your/video.mp4 --output results/car_video_segmentation/ --fps 10
    ```

4.  **Using larger models for potentially higher accuracy (requires more GPU VRAM):**
    ```bash
    sowlv2-detect --prompt "dog" --input dog.jpg --output results_large/ \
                  --owl-model "google/owlv2-large-patch14-ensemble" \
                  --sam-model "facebook/sam2.1-hiera-large"
    ```

### Output Structure:

The tool saves results in the specified output directory. For each detected object, SOWLv2 generates:
*   A **binary mask** image (e.g., `imagename_object0_mask.png`): Grayscale PNG where foreground pixels are white (255) and background pixels are black (0).
*   An **overlay image** (e.g., `imagename_object0_overlay.png`): The original image with the segmentation mask overlaid (typically colored with transparency).

Objects are numbered sequentially (e.g., `object0`, `object1`) in the order they are detected. For video inputs, output filenames will also include frame identifiers, and separate videos for each object's masks and overlays will be generated (e.g., `obj0_mask_video.mp4`, `obj0_overlay_video.mp4`).

### <a name="configuration"></a>Configuration File (Optional):

You can use a YAML configuration file to specify arguments, which is useful for managing complex settings or reproducing experiments.

Example `config.yaml`:
```yaml
prompt: "a pedestrian crossing the street"
input: "data/street_scene.mp4"
output: "results/pedestrian_video"
owl-model: "google/owlv2-base-patch16-ensemble"
sam-model: "facebook/sam2.1-hiera-small"
threshold: 0.15
fps: 15
device: "cuda"
```

Run with config:
```bash
sowlv2-detect --config config.yaml
```
Note: Command-line arguments will override values specified in the config file if both are provided (except for `prompt` and `input` which will be taken from config if not given on CLI).

## 🧠 How It Works

SOWLv2 follows a two-stage pipeline:

1.  **OWLv2 Detection:**
    The input image/frame is processed using the specified OWLv2 model (default: `google/owlv2-base-patch16-ensemble`) via the Hugging Face `transformers` library. Based on the provided text prompt, OWLv2 identifies relevant objects and outputs their bounding boxes along with confidence scores.

2.  **SAM 2 Segmentation:**
    For each bounding box detected by OWLv2 (above the specified confidence threshold), the SAM 2 model (default: `facebook/sam2.1-hiera-small`) is invoked. SAM 2 takes the original image and the bounding box as input prompts to generate a precise segmentation mask for the object within that box. For videos, SAM 2's video-specific capabilities are used to propagate masks across frames.

3.  **Output Generation:**
    The generated binary masks are saved as grayscale PNG files. Additionally, these masks are overlaid onto the original images/frames to create visually inspectable results, which are also saved. For video inputs, individual frame outputs are processed, and then compiled into per-object mask videos and overlay videos.

## 📦 Dependencies

SOWLv2 relies on the following major Python packages:
*   `torch` (PyTorch)
*   `transformers` (for OWLv2 models)
*   `sam2` (Meta’s SAM 2 package)
*   `opencv-python` (for image and video processing)
*   `numpy`, `Pillow`, `pyyaml`, `huggingface_hub`

These dependencies are listed in `setup.py` and `requirements.txt` and will be installed automatically when using pip.

## 📜 License

This project is licensed under the Apache 2.0 License.
See the `LICENSE` file for more details.

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please feel free to fork the repository, make your changes, and submit a pull request. For major changes or new features, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

SOWLv2 builds upon the incredible work of researchers and developers behind these foundational models and libraries:
*   **OWLv2:** [Google Research](https://huggingface.co/docs/transformers/en/model_doc/owlv2)
    *   Paper: [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)
*   **SAM 2:** [Meta AI Research](https://github.com/facebookresearch/sam2)
    *   Paper: [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)

We extend our sincere gratitude to the authors and maintainers for open-sourcing their code and models.

---
### 🌟 Developed by Csaba Bolyòs 🚀

Connect with me:
[🔗 LinkedIn](https://www.linkedin.com/in/csaba-boly%C3%B2s-00a11767/) | [📓 Google Colab Demo](https://colab.research.google.com/drive/1vX6P4KNmWoisY-Vfq6bAVunsHaLrC-AO)
```

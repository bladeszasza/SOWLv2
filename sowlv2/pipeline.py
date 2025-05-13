import os
import subprocess
import shutil
import tempfile
import torch
import numpy as np
from PIL import Image
from sowlv2 import video_utils
from sowlv2.owl import OWLV2Wrapper
from sowlv2.sam2_wrapper import SAM2Wrapper
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SOWLv2Pipeline:
    
    def __init__(self, owl_model, sam_model, threshold=0.4, fps=24, device="cuda", owl_skip_frames=3):
        """Initializes the pipeline with models and parameters."""
        self.device_type = torch.device(device).type # Store 'cuda' or 'cpu'
        logging.info(f"Initializing OWLV2Wrapper with model: {owl_model} on device type: {self.device_type}")
        self.owl = OWLV2Wrapper(model_name=owl_model, device=device)
        logging.info(f"Initializing SAM2Wrapper with model: {sam_model} on device type: {self.device_type}")
        self.sam = SAM2Wrapper(model_name=sam_model, device=device)
        self.threshold = threshold
        self.fps = fps
        self.owl_skip_frames = owl_skip_frames
        logging.info(f"Pipeline initialized on device type: {self.device_type} with threshold: {self.threshold}")

    def _process_with_autocast(self, func, *args, **kwargs):
        """Wrapper to run a processing function under a float32 autocast context if on CUDA."""
        if self.device_type == "cuda":
            with torch.autocast(device_type=self.device_type, dtype=torch.float32, enabled=True):
                logging.debug(f"Running {func.__name__} under CUDA float32 autocast.")
                return func(*args, **kwargs)
        else:
            logging.debug(f"Running {func.__name__} without explicit autocast (device: {self.device_type}).")
            return func(*args, **kwargs)

    def _process_image_core(self, image_path, prompt, output_dir):
        """Core logic for single image processing."""
        image = Image.open(image_path).convert("RGB")
        detections = self.owl.detect(image, prompt, self.threshold)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if not detections:
            logging.warning(f"No objects detected for prompt '{prompt}' in image '{image_path}'.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        for idx, det in enumerate(detections):
            box = det["box"]
            mask = self.sam.segment(image, box)
            if mask is None:
                logging.debug(f"SAM returned no mask for a detection in {image_path}, obj {idx}.")
                continue
            
            mask_binary = (mask > 0).astype(np.uint8)
            mask_img = Image.fromarray(mask_binary * 255).convert("L")
            mask_file = os.path.join(output_dir, f"{base_name}_object{idx}_mask.png")
            mask_img.save(mask_file)
            
            overlay = self._create_overlay(image, mask_binary)
            overlay_file = os.path.join(output_dir, f"{base_name}_object{idx}_overlay.png")
            overlay.save(overlay_file)
        logging.info(f"Processed image {image_path} for prompt '{prompt}'. Outputs in {output_dir}")

    def process_image(self, image_path, prompt, output_dir):
        self._process_with_autocast(self._process_image_core, image_path, prompt, output_dir)

    def _process_frames_core(self, folder_path, prompt, output_dir):
        files = sorted(os.listdir(folder_path))
        processed_any = False
        for fname in files:
            infile = os.path.join(folder_path, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                continue
            self._process_image_core(infile, prompt, output_dir)
            processed_any = True
        if not processed_any:
            logging.warning(f"No processable image files found in folder {folder_path}.")
        else:
            logging.info(f"Finished processing frames in {folder_path} for prompt '{prompt}'. Outputs in {output_dir}")

    def process_frames(self, folder_path, prompt, output_dir):
        self._process_with_autocast(self._process_frames_core, folder_path, prompt, output_dir)

    def _process_video_core(self, video_path, prompt, output_dir):
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="sowlv2_frames_")
            logging.info(f"Created temporary directory for frames: {tmp_dir}")

            ffmpeg_cmd = [
                "ffmpeg", "-i", video_path, "-r", str(self.fps), "-q:v", "2",
                os.path.join(tmp_dir, "%06d.jpg"), "-hide_banner", "-loglevel", "warning"
            ]
            logging.info(f"Running ffmpeg to extract frames at {self.fps} FPS...")
            subprocess.run(ffmpeg_cmd, check=True)
            logging.info("Frame extraction complete.")

            frame_files = sorted([f for f in os.listdir(tmp_dir) if f.lower().endswith(".jpg")])
            if not frame_files:
                logging.warning("No frames extracted. Aborting.")
                return

            logging.info("Initializing SAM2 video state...")
            state = self.sam.init_state(tmp_dir)
            logging.info("SAM2 state initialized.")

            obj_id_counter = 1
            total_detections_added = 0
            owl_run_interval = self.owl_skip_frames + 1
            logging.info(f"Starting OWLv2 detection for prompt '{prompt}' on {len(frame_files)} frames, OWLv2 will run every {owl_run_interval} frame(s).")

            for frame_idx, frame_filename in enumerate(frame_files):
                frame_path = os.path.join(tmp_dir, frame_filename)
                try:
                    current_sam_frame_idx = frame_idx
                    if frame_idx % owl_run_interval == 0:
                        logging.debug(f"Processing frame {frame_idx+1}/{len(frame_files)} ({frame_filename}) for OWLv2.")
                        image = Image.open(frame_path).convert("RGB")
                        detections = self.owl.detect(image, prompt, self.threshold)

                        if detections:
                            logging.debug(f"  Found {len(detections)} potential objects in frame {frame_idx+1}.")
                            for det in detections:
                                box = det["box"]
                                logging.debug(f"    Adding Box: {box} with tentative obj_id: {obj_id_counter}")
                                self.sam.add_new_box(state=state, frame_idx=current_sam_frame_idx, box=box, obj_idx=obj_id_counter)
                                obj_id_counter += 1
                                total_detections_added += 1
                        else:
                            logging.debug(f"  No objects found for prompt '{prompt}' in frame {frame_idx+1}.")
                    else:
                        logging.debug(f"Skipping OWLv2 detection for frame {frame_idx+1}/{len(frame_files)}.")
                except Exception as e:
                    logging.error(f"Error processing frame {frame_filename} during detection: {e}", exc_info=True)
                    continue

            logging.info(f"Finished OWLv2 detection. Added {total_detections_added} boxes using {obj_id_counter-1} unique IDs.")
            if total_detections_added == 0:
                logging.warning(f"No objects matching '{prompt}' detected in any frame. Aborting segmentation.")
                return

            logging.info("Starting SAM2 mask propagation across video...")
            propagation_results = []
            try:
                for fidx, current_obj_ids, current_mask_logits in self.sam.propagate_in_video(state):
                    propagation_results.append((fidx, current_obj_ids, current_mask_logits))
            except Exception as e:
                 logging.error(f"Error during SAM2 propagation: {e}", exc_info=True)
                 return
            logging.info(f"SAM2 propagation finished. Processing {len(propagation_results)} frames with masks.")
            
            os.makedirs(output_dir, exist_ok=True)
            if not propagation_results:
                logging.warning("No propagation results from SAM2. No masks will be saved.")
                return

            logging.info("Saving individual frame masks and overlays...")
            for fidx, current_obj_ids, current_mask_logits in propagation_results:
                frame_num_for_filename = fidx + 1 
                frame_filename_jpg = f"{frame_num_for_filename:06d}.jpg"
                frame_path = os.path.join(tmp_dir, frame_filename_jpg)

                if not os.path.exists(frame_path):
                     logging.warning(f"Frame file {frame_path} not found during saving. Skipping.")
                     continue
                try:
                    img = Image.open(frame_path).convert("RGB")
                    self._video_save_masks_and_overlays(img, frame_num_for_filename, current_obj_ids, current_mask_logits, output_dir)
                except Exception as e:
                    logging.error(f"Error saving masks/overlays for frame index {fidx} ({frame_filename_jpg}): {e}", exc_info=True)
            logging.info(f"✅ Individual frame processing finished; results in {output_dir}")

            output_fps_val = self.fps
            logging.info(f"Generating per-object mask and overlay videos at {output_fps_val} FPS...")
            video_utils.generate_per_object_videos(output_dir, fps=output_fps_val)
            logging.info(f"✅ Per-object video generation finished; results in {output_dir}")

        except FileNotFoundError as e:
            logging.error(f"File not found error during video processing: {e}", exc_info=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"ffmpeg command failed: {e}", exc_info=True)
        except Exception as e:
            logging.error(f"An unexpected error occurred during video processing: {e}", exc_info=True)
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                    logging.info(f"Removed temporary directory: {tmp_dir}")
                except Exception as e:
                    logging.error(f"Failed to remove temporary directory {tmp_dir}: {e}", exc_info=True)

    def process_video(self, video_path, prompt, output_dir):
        self._process_with_autocast(self._process_video_core, video_path, prompt, output_dir)
            
    def _video_save_masks_and_overlays(self, pil_img, frame_idx, obj_ids_list, masks_logits, out_dir):
        """Process and store masks and overlays for video generation.
        obj_ids_list is expected to be a Python list of object IDs.
        masks_logits is expected to be a torch.Tensor [num_objects, H, W].
        """
        base = f"{frame_idx:06d}"
        
        if not isinstance(masks_logits, torch.Tensor):
            logging.error(f"Masks_logits for frame {frame_idx} is not a tensor, type: {type(masks_logits)}. Skipping save for this frame.")
            return
        
        if not isinstance(obj_ids_list, list):
            logging.error(f"obj_ids for frame {frame_idx} is not a list, type: {type(obj_ids_list)}. Skipping save for this frame.")
            return

        if masks_logits.ndim == 2: # If it's a single mask [H,W] for a single object
            masks_logits = masks_logits.unsqueeze(0) # Add batch dim for objects: [1, H, W]
        
        num_masks = masks_logits.shape[0]
        num_obj_ids = len(obj_ids_list)

        if num_masks != num_obj_ids:
            logging.error(f"Mismatch between number of masks ({num_masks}) and obj_ids ({num_obj_ids}) for frame {frame_idx}. Skipping.")
            return

        for i in range(num_obj_ids):
            obj_id = obj_ids_list[i] # Directly use the item from the list
            # Ensure obj_id is an int if it's a tensor element (e.g. tensor(5))
            if isinstance(obj_id, torch.Tensor):
                obj_id = obj_id.item()
            elif not isinstance(obj_id, (int, np.integer)): # Check if it's already a Python or NumPy int
                try:
                    obj_id = int(obj_id) # Try to convert to int
                except ValueError:
                    logging.error(f"obj_id '{obj_id}' (type: {type(obj_id)}) for frame {frame_idx} cannot be converted to int. Skipping object.")
                    continue


            mask_logit = masks_logits[i]  # Get the logit for the current object [H,W]
            
            mask_prob = torch.sigmoid(mask_logit) 
            mask_binary_np = (mask_prob > 0.5).cpu().numpy().astype(np.uint8) 

            if mask_binary_np.ndim != 2:
                logging.error(f"mask_binary_np for obj {obj_id} in frame {frame_idx} has unexpected ndim: {mask_binary_np.ndim}. Skipping object.")
                continue

            mask_pil = Image.fromarray(mask_binary_np * 255).convert("L")
    
            mask_path = os.path.join(out_dir, f"{base}_obj{obj_id}_mask.png")
            mask_pil.save(mask_path)
    
            overlay = self._create_overlay(pil_img, mask_binary_np)
            overlay_path = os.path.join(out_dir, f"{base}_obj{obj_id}_overlay.png")
            overlay.save(overlay_path)
        # No need to return mask_frames, overlay_frames if not used
       
    def _create_overlay(self, image, mask_numpy): 
        """
        Blend a red mask with the input PIL.Image and return the overlay as a PIL.Image.
        `mask_numpy` should be a 2D NumPy array (binary 0 or 1, or 0 and 255).
        """
        if not isinstance(mask_numpy, np.ndarray) or mask_numpy.ndim != 2:
            raise ValueError(f"Expected 2-D NumPy mask, got type {type(mask_numpy)} with shape {mask_numpy.shape if hasattr(mask_numpy, 'shape') else 'N/A'}")
    
        image_np   = np.asarray(image.convert("RGB"), dtype=np.uint8)
        overlay_np = image_np.copy()
    
        bool_mask = (mask_numpy > 0) 
    
        red = np.array([255, 0, 0], dtype=np.uint8)
        
        overlay_np[bool_mask] = (
            0.5 * overlay_np[bool_mask] + 0.5 * red
        ).astype(np.uint8)
    
        return Image.fromarray(overlay_np)
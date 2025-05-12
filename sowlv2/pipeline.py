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
    
    def __init__(self, owl_model, sam_model, threshold=0.4, fps=24, device="cuda"):
        """Initializes the pipeline with models and parameters."""
        logging.info(f"Initializing OWLV2Wrapper with model: {owl_model}")
        self.owl = OWLV2Wrapper(model_name=owl_model, device=device)
        logging.info(f"Initializing SAM2Wrapper with model: {sam_model}")
        self.sam = SAM2Wrapper(model_name=sam_model, device=device)
        self.threshold = threshold
        self.fps = fps # Used for frame extraction
        self.device = device
        logging.info(f"Pipeline initialized on device: {self.device} with threshold: {self.threshold}")

    def process_image(self, image_path, prompt, output_dir):
        """Process a single image file."""
        image = Image.open(image_path).convert("RGB")
        detections = self.owl.detect(image, prompt, self.threshold)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        if not detections:
            print(f"No objects detected for prompt '{prompt}' in image '{image_path}'.")
        for idx, det in enumerate(detections):
            box = det["box"]  # [x1, y1, x2, y2]
            # Run SAM segmentation on the detected box
            mask = self.sam.segment(image, box)
            if mask is None:
                continue
            # Save binary mask
            mask_img = Image.fromarray(mask * 255).convert("L")
            mask_file = os.path.join(output_dir, f"{base_name}_object{idx}_mask.png")
            mask_img.save(mask_file)
            # Create and save overlay image
            overlay = self._create_overlay(image, mask)
            overlay_file = os.path.join(output_dir, f"{base_name}_object{idx}_overlay.png")
            overlay.save(overlay_file)

    def process_frames(self, folder_path, prompt, output_dir):
        """Process a folder of images (frames)."""
        files = sorted(os.listdir(folder_path))
        for fname in files:
            infile = os.path.join(folder_path, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                continue
            self.process_image(infile, prompt, output_dir)
            
    def process_video(self, video_path, prompt, output_dir):
        """
        Process a single video file. Performs object detection on *each frame*
        and uses SAM2 to segment and propagate masks based on all detections.
        """
        tmp_dir = None # Initialize tmp_dir to ensure it's available in finally block
        try:
            tmp_dir = tempfile.mkdtemp(prefix="sowlv2_frames_")
            logging.info(f"Created temporary directory for frames: {tmp_dir}")

            # --- 1. Extract Frames ---
            ffmpeg_cmd = [
                "ffmpeg", "-i", video_path, "-r", str(self.fps),
                os.path.join(tmp_dir, "%06d.jpg"), "-hide_banner", "-loglevel", "warning" # Changed loglevel
            ]
            logging.info(f"Running ffmpeg to extract frames at {self.fps} FPS...")
            subprocess.run(ffmpeg_cmd, check=True)
            logging.info("Frame extraction complete.")

            frame_files = sorted([f for f in os.listdir(tmp_dir) if f.lower().endswith(".jpg")])
            if not frame_files:
                logging.warning("No frames extracted. Aborting.")
                return

            # --- 2. Initialize SAM2 State ---
            logging.info("Initializing SAM2 video state...")
            state = self.sam.init_state(tmp_dir)
            logging.info("SAM2 state initialized.")

            # --- 3. Detect Objects Frame-by-Frame and Add to SAM2 State ---
            obj_id_counter = 1
            total_detections_added = 0
            logging.info(f"Starting OWLv2 detection for prompt '{prompt}' on {len(frame_files)} frames...")

            for frame_idx, frame_filename in enumerate(frame_files):
                frame_path = os.path.join(tmp_dir, frame_filename)
                try:
                    # Ensure frame_idx matches the 0-based index SAM expects
                    current_sam_frame_idx = frame_idx

                    logging.debug(f"Processing frame {frame_idx+1}/{len(frame_files)} ({frame_filename})")
                    image = Image.open(frame_path).convert("RGB")

                    # Run OWLv2 detection on the current frame
                    detections = self.owl.detect(image, prompt, self.threshold)

                    if detections:
                        logging.debug(f"  Found {len(detections)} potential objects in frame {frame_idx+1}.")
                        for det in detections:
                            # Handle potential multiple boxes per detection if OWLv2 returns structured results differently
                            # Assuming det['box'] is [x1, y1, x2, y2]
                            box = det["box"]
                            score = det["score"]
                            logging.debug(f"    Adding Box: {box} (Score: {score:.2f}) with tentative obj_id: {obj_id_counter}")

                            # Add the detected box to SAM2 state for this specific frame
                            # We assume add_new_box handles associating this box with the object ID
                            # across frames if necessary internal logic exists in SAM2,
                            # otherwise, it treats each add as a potential new track seed.
                            _, _, _ = self.sam.add_new_box(
                                state=state,
                                frame_idx=current_sam_frame_idx, # Use 0-based index
                                box=box,
                                obj_idx=obj_id_counter
                            )
                            obj_id_counter += 1
                            total_detections_added += 1
                    else:
                         logging.debug(f"  No objects found for prompt '{prompt}' in frame {frame_idx+1}.")

                except Exception as e:
                    logging.error(f"Error processing frame {frame_filename}: {e}", exc_info=True)
                    # Decide if you want to continue or abort on frame error
                    continue # Continue to next frame

            logging.info(f"Finished OWLv2 detection. Added {total_detections_added} boxes across all frames using {obj_id_counter-1} unique IDs.")

            if total_detections_added == 0:
                logging.warning(f"No objects matching '{prompt}' detected in any frame. Aborting segmentation.")
                return # Exit early if no detections were ever added

            # --- 4. Propagate Masks using SAM2 ---
            logging.info("Starting SAM2 mask propagation across video...")
            propagation_results = []
            try:
                for fidx, obj_ids, mask_logits in self.sam.propagate_in_video(state):
                    propagation_results.append((fidx, obj_ids, mask_logits))
            except Exception as e:
                 logging.error(f"Error during SAM2 propagation: {e}", exc_info=True)
                 # Handle propagation error, maybe save partial results or abort
                 return # Abort for now if propagation fails

            logging.info(f"SAM2 propagation finished. Processing {len(propagation_results)} frames with masks.")


            # --- 5. Save Masks and Overlays for each frame ---
            # This loop now iterates through the results of the propagation
            logging.info("Saving individual frame masks and overlays...")
            for fidx, obj_ids, mask_logits in propagation_results:
                # SAM's propagate_in_video likely returns 0-based fidx
                # corresponding to the frame list order.
                # Our frame files are named %06d starting from 1.
                frame_num_for_filename = fidx + 1
                frame_filename = f"{frame_num_for_filename:06d}.jpg"
                frame_path = os.path.join(tmp_dir, frame_filename)

                if not os.path.exists(frame_path):
                     logging.warning(f"Frame file {frame_path} not found during saving. Skipping.")
                     continue

                try:
                    img = Image.open(frame_path).convert("RGB")
                    # Note: _video_save_masks_and_overlays saves individual pngs
                    self._video_save_masks_and_overlays(img, frame_num_for_filename, obj_ids, mask_logits, output_dir)
                except Exception as e:
                    logging.error(f"Error saving masks/overlays for frame index {fidx} ({frame_filename}): {e}", exc_info=True)
                    # Continue to next frame even if one fails saving

            logging.info(f"✅ Individual frame processing finished; results in {output_dir}")

            # --- 6. Generate Per-Object Videos ---
            # Use the actual frame rate used for extraction if possible, default fallback
            output_fps = self.fps
            logging.info(f"Generating per-object mask and overlay videos at {output_fps} FPS...")
            video_utils.generate_per_object_videos(output_dir, fps=output_fps)
            logging.info(f"✅ Per-object video generation finished; results in {output_dir}")

        except FileNotFoundError as e:
            logging.error(f"File not found error during video processing: {e}", exc_info=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"ffmpeg command failed: {e}", exc_info=True)
        except Exception as e:
            # Catch any other unexpected errors during the process
            logging.error(f"An unexpected error occurred during video processing: {e}", exc_info=True)
        finally:
            # --- 7. Cleanup Temporary Directory ---
            if tmp_dir and os.path.exists(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                    logging.info(f"Removed temporary directory: {tmp_dir}")
                except Exception as e:
                    logging.error(f"Failed to remove temporary directory {tmp_dir}: {e}", exc_info=True)


    def _save_masks_and_overlays(self, pil_img, frame_idx, obj_ids, masks, out_dir):
        """Write <frame>_obj<id>_mask.png and _overlay.png files."""
        base = f"{frame_idx:06d}"
        for obj_id, mask in zip(obj_ids, masks):
            mask_bin = ((mask > 0.5).astype(np.uint8)) * 255
            Image.fromarray(mask_bin).save(
                os.path.join(out_dir, f"{base}_obj{obj_id}_mask.png")
            )
            overlay = self._create_overlay(pil_img, mask > 0.5)
            overlay.save(
                os.path.join(out_dir, f"{base}_obj{obj_id}_overlay.png")
            )
            
    def _video_save_masks_and_overlays(self, pil_img, frame_idx, obj_ids, masks, out_dir):
        """Process and store masks and overlays for video generation."""
        base = f"{frame_idx:06d}"
        mask_frames = []
        overlay_frames = []
    
        for obj_id, mask in zip(obj_ids, masks):
            # Convert mask to binary
            mask_bin = ((mask > 0.5).cpu().numpy().astype(np.uint8)) * 255
            mask_bin = np.squeeze(mask_bin)  # Removes dimensions of size 1

            # Ensure mask_bin is 2D
            if mask_bin.ndim != 2:
                raise ValueError(f"mask_bin has unexpected number of dimensions: {mask_bin.ndim}")

            mask_pil = Image.fromarray(mask_bin)
    
            # Save individual mask image
            mask_path = os.path.join(out_dir, f"{base}_obj{obj_id}_mask.png")
            mask_pil.save(mask_path)
    
            # Create and save overlay image
            overlay = self._create_overlay(pil_img, mask > 0.5)
            overlay_path = os.path.join(out_dir, f"{base}_obj{obj_id}_overlay.png")
            overlay.save(overlay_path)
    
            # Store frames for video
            mask_frames.append(mask_pil)
            overlay_frames.append(overlay)
        return mask_frames, overlay_frames
       
    def _create_overlay(self, image, mask):
        """
        Blend a red mask with the input PIL.Image and return the overlay as a PIL.Image.
        Works whether `mask` is NumPy or a PyTorch tensor and regardless of
        leading singleton dimensions.
        """
        # 1. Make sure we have a NumPy binary mask on CPU and squeeze extra dims
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()          # move off GPU
        mask = np.squeeze(mask)                         # drop dimensions of size 1
        if mask.ndim != 2:
            raise ValueError(f"Expected 2-D mask, got shape {mask.shape}")
    
        # 2. Prepare image & output
        image_np   = np.asarray(image, dtype=np.uint8)
        overlay_np = image_np.copy()
    
        # 3. Boolean index with a 2-D mask – NumPy broadcasts the channel dim
        red = np.array([255, 0, 0], dtype=np.uint8)
        overlay_np[mask > 0] = (
            0.5 * overlay_np[mask > 0] + 0.5 * red
        ).astype(np.uint8)
    
        return Image.fromarray(overlay_np)

    

"""File system utility functions for SOWLv2."""
import os

def remove_empty_folders(root_dir: str) -> None:
    """
    Recursively remove all empty folders under the given root directory.
    Args:
        root_dir (str): The root directory to start searching for empty folders.
    """
    for dirpath, dirnames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            remove_empty_folders(full_path)

    if not os.listdir(root_dir):
        try:
            os.rmdir(root_dir)
        except OSError as e:
            print(f"Failed to remove {root_dir}: {e}")

def create_output_directories(base_dir: str, include_video: bool = False) -> dict:
    """Create a standardized directory structure for pipeline outputs.

    Args:
        base_dir: Base directory for outputs
        include_video: Whether to create video-specific directories

    Returns:
        Dictionary mapping directory types to their paths
    """
    dirs = {
        "binary": os.path.join(base_dir, "binary"),
        "binary_frames": os.path.join(base_dir, "binary", "frames"),
        "binary_merged": os.path.join(base_dir, "binary", "merged"),
        "overlay": os.path.join(base_dir, "overlay"),
        "overlay_frames": os.path.join(base_dir, "overlay", "frames"),
        "overlay_merged": os.path.join(base_dir, "overlay", "merged")
    }

    if include_video:
        dirs.update({
            "video": os.path.join(base_dir, "video"),
            "video_binary": os.path.join(base_dir, "video", "binary"),
            "video_overlay": os.path.join(base_dir, "video", "overlay")
        })

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs

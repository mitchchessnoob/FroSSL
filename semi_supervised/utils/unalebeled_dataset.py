import os
import shutil

def flatten_image_directory(source_dir, target_dir):
    """
    Flattens a folder structure by moving all images from subfolders into a single directory.

    Args:
        source_dir (str): Path to the source directory with class subfolders.
        target_dir (str): Path to the target directory where all images will be stored.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # Create target directory if it doesn't exist

    # Traverse the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check if the file is an image (based on extension)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_dir, file)

                # Handle potential filename conflicts by renaming duplicates
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                        counter += 1

                # Copy the file to the target directory
                shutil.copy2(source_path, target_path)

    print(f"All images have been copied to {target_dir}.")

# Example usage
source_directory = "/content/office31/Office31/Office31_amazon" # Replace with the path to your structured folder
target_directory = "/content/office31/Office31/Office31_amazon_flat"  # Replace with the path to your target folder
flatten_image_directory(source_directory, target_directory)

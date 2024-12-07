import kagglehub
import shutil
def dataset():
  # Download latest version
  path = kagglehub.dataset_download("mei1963/office31")


  destination_path = './1'
  shutil.rmtree(destination_path)
  # Move the folder
  shutil.move(path, destination_path)

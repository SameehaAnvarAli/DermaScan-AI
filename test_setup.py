import torch
import cv2
import pandas as pd
from PIL import Image

print("DermaScan AI setup test")
print("-----------------------")

print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("Pandas:", pd.__version__)

print("GPU available:", torch.cuda.is_available())

print("All basic libraries are working!")
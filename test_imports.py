from PIL import Image
from tqdm import tqdm
import torch
import torchvision

print("PIL:", Image.__version__)
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)

for i in tqdm(range(5)):
    pass

print("Everything is working!")
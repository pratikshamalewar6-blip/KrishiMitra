# pyrefly: ignore [missing-import]
from PIL import Image

from data.transforms import train_transform

image = Image.new(
    "RGB",
    (500, 400),
    color="green",
)

tensor = train_transform(image)

print(type(tensor))
print(tensor.shape)
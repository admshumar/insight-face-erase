from PIL import Image
from rectangle import Mask
import torch

print(torch.__version__)

m = Mask(256, 256, [0, 0, 128, 128])

img = Image.fromarray(m.array)
img.show()

for value in m.__dict__.keys():
    print(value)
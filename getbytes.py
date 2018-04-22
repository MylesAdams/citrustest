import os
import io
from PIL import Image
from array import array

with open("kid.jpg", "rb") as imageFile:
    f = imageFile.read()
    b = bytearray(f)

print(b[0])
file_object = open('kid_bytes.txt', 'w')

file_object.write(str(b))

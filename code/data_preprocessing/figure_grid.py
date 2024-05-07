from PIL import Image
import os
from PIL import ImageEnhance
import numpy as np

# Path to the folder containing images
folder_path = "./figs/wins"

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

# Define the dimensions of the grid (e.g., 3x3 grid)
grid_width = 5
grid_height = 14

# Open the first image to get its width and height
with Image.open(os.path.join(folder_path, image_files[0])) as first_image:
    image_width, image_height = first_image.size

# Create a function to scale images to the same size
def scale_image(image:Image.Image, width, height):
    aspect_ratio = float(image.width) / float(image.height)
    new_height = int(height)
    new_width = int(width)
    return image.resize((new_width, new_height), Image.ANTIALIAS)

# Create a blank image to hold the grid
grid_image = Image.new('RGB', (grid_width * image_width, grid_height * image_height),color='white')

# Loop through the images and paste them onto the grid
for i, image_file in enumerate(image_files):
    row = i // grid_width
    col = i % grid_width
    img = Image.open(os.path.join(folder_path, image_file))
    img = scale_image(img, image_width, image_height)
    grid_image.paste(img, (col * image_width, row * image_height))

# Save the final grid image
grid_image.save("grid_image.jpg")
import os
import random
from PIL import Image

import random

'''
Defines a random 128 x 128 patch of the background image to crop out
'''
def crop_random_patch(img, patch_size=(128, 128)):
    # Defines width, height, patch height, patch width
    w, h = img.size
    pw, ph = patch_size
    # Condition if either background image dimension is less than 128
    if w < pw or h < ph:
        raise ValueError("Background image is smaller than the patch size!")
    
    # Selects a random 128 x 128 area of the background images and crops it
    x = random.randint(0, w - pw)
    y = random.randint(0, h - ph)
    return img.crop((x, y, x + pw, y + ph))

'''
Pastes the foreground image (the vehicle without background) onto a 128 x 128 background
'''
def paste_foreground_on_background(foreground_path, background_path, output_path, resize_range=(0.5, 0.7), patch_size=(128,128), max_retries=10):
    # Defines the foreground and the background
    fg = Image.open(foreground_path).convert("RGBA")
    bg = Image.open(background_path).convert("RGBA")

    # Crop a random 64x64 patch from background
    bg_patch = crop_random_patch(bg, patch_size)

    # Resize foreground to fit inside patch size (without padding)
    fg.thumbnail(patch_size, Image.Resampling.LANCZOS)

    # Scale foreground randomly (smaller than patch size)
    for _ in range(max_retries):
        scale = random.uniform(*resize_range)
        new_size = (int(fg.width * scale), int(fg.height * scale))
        fg_resized = fg.resize(new_size, Image.Resampling.LANCZOS)

        max_x = patch_size[0] - fg_resized.width
        max_y = patch_size[1] - fg_resized.height

        if max_x >= 0 and max_y >= 0:
            fg = fg_resized
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)
            break
    else:
        print(f"Could not fit foreground {foreground_path} on background patch {background_path} after {max_retries} tries")
        return

    if random.random() > 0.5:
        fg = fg.transpose(Image.FLIP_LEFT_RIGHT)

    bg_patch.paste(fg, (x, y), fg)
    bg_patch = bg_patch.convert("RGB")
    bg_patch.save(output_path)

# Defines the foreground, background, and output directories. This system is very robust as you can change
# the folders easily to customize the generated data (e.g., change foreground_dir to ...createdata\train\car, 
# change background_dir to ...createdata\train\unsplash_backgrounds, and output_dir to 
# ...customdata\train\car to generate new custom car data in that folder).
foreground_dir = r'C:\Users\bluet\OneDrive\Documents\custom-cnn-for-cifar-10\createdata\test\pickup_truck'
background_dir = r'C:\Users\bluet\OneDrive\Documents\custom-cnn-for-cifar-10\createdata\test\unsplash_backgrounds'
output_dir = r'C:\Users\bluet\OneDrive\Documents\custom-cnn-for-cifar-10\customdata\test\pickup_truck'
os.makedirs(output_dir, exist_ok=True)

foregrounds = [os.path.join(foreground_dir, f) for f in os.listdir(foreground_dir) if f.lower().endswith('.png')]
backgrounds = [os.path.join(background_dir, f) for f in os.listdir(background_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

bg_size = (128, 128)  # define background size globally

# 500 per class for training 150 per class for testing
num_images = 150
for i in range(num_images):
    fg_path = random.choice(foregrounds)
    bg_path = random.choice(backgrounds)
    out_path = os.path.join(output_dir, f"synthetic_{i}.jpg")
    try:
        paste_foreground_on_background(fg_path, bg_path, out_path, patch_size=bg_size)
    except ValueError as e:
        print(f"Skipping background {bg_path} due to error: {e}")
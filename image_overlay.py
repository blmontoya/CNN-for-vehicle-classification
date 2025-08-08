import os
import random
from PIL import Image

def paste_foreground_on_background(foreground_path, background_path, output_path, resize_range=(0.2, 0.5)):
    # Load images
    fg = Image.open(foreground_path).convert("RGBA")
    bg = Image.open(background_path).convert("RGBA")

    # Resize foreground randomly (keep aspect ratio)
    scale = random.uniform(*resize_range)
    new_size = (int(fg.width * scale), int(fg.height * scale))
    fg = fg.resize(new_size, Image.ANTIALIAS)

    # Optional random horizontal flip
    if random.random() > 0.5:
        fg = fg.transpose(Image.FLIP_LEFT_RIGHT)

    # Random position for pasting
    max_x = bg.width - fg.width
    max_y = bg.height - fg.height
    if max_x < 0 or max_y < 0:
        print(f"Foreground too large for background: {foreground_path}")
        return

    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Paste foreground onto background
    bg.paste(fg, (x, y), fg)  # `fg` is used as its own alpha mask

    # Save output
    bg = bg.convert("RGB")  # Remove alpha for saving as JPEG
    bg.save(output_path)

# Paths
foreground_dir = 'foregrounds'
background_dir = 'backgrounds'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Load file names
foregrounds = [os.path.join(foreground_dir, f) for f in os.listdir(foreground_dir) if f.endswith('.png')]
backgrounds = [os.path.join(background_dir, f) for f in os.listdir(background_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Generate synthetic images
num_images = 100

for i in range(num_images):
    fg_path = random.choice(foregrounds)
    bg_path = random.choice(backgrounds)
    out_path = os.path.join(output_dir, f"synthetic_{i}.jpg")
    paste_foreground_on_background(fg_path, bg_path, out_path)
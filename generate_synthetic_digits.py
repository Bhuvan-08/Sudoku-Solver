import cv2
import numpy as np
import os
import random

# Create output folder if it doesn't exist
output_dir = "synthetic_digits"
os.makedirs(output_dir, exist_ok=True)

# Fonts to use
fonts = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX
]

# Generate 50 images per digit per font
for digit in range(10):
    for font in fonts:
        for i in range(50):
            # Create blank 28x28 image
            img = np.zeros((28, 28), dtype=np.uint8)

            # Generate random thickness and position
            font_scale = random.uniform(0.8, 1.2)
            thickness = random.randint(1, 2)

            # Calculate size to center text
            text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]
            text_x = (28 - text_size[0]) // 2
            text_y = (28 + text_size[1]) // 2

            # Draw digit on image
            cv2.putText(img, str(digit), (text_x, text_y), font, font_scale, 255, thickness)

            # Save image
            filename = f"{output_dir}/{digit}_{font}_{i}.png"
            cv2.imwrite(filename, img)

print("✅ 2500 synthetic digit images saved in 'synthetic_digits/'")
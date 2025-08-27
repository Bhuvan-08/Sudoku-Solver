import os
import cv2
import numpy as np

# Create output directory if it doesn't exist
os.makedirs("synthetic_digits/0_blank", exist_ok=True)

# Generate 300 blank cells with light random noise
for i in range(300):
    blank = np.zeros((28, 28), dtype=np.uint8)
    noise = np.random.randint(0, 10, (28, 28), dtype=np.uint8)  # Add light pixel noise
    noisy_blank = cv2.add(blank, noise)
    cv2.imwrite(f"synthetic_digits/0_blank/0_blank_{i}.png", noisy_blank)

print("✅ Generated 300 blank images in synthetic_digits/0_blank/")

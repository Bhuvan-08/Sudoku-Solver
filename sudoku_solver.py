import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import shutil 

# --- Configuration Parameters ---
MODEL_PATH = 'improved_digit_cnn.h5' # Your specified model name
DEBUG_MODE = True # Set to True to save debug images of cell processing
CONFIDENCE_THRESHOLD = 0.75 # Standard confidence threshold for predictions

# --- Image Preprocessing Parameters ---
GAUSSIAN_BLUR_KERNEL = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 11
ADAPTIVE_THRESH_C = 2
MORPH_OPEN_KERNEL = (2, 2) # Restored
MORPH_OPEN_ITERATIONS = 1 # Restored

# --- Cell Extraction Parameters ---
CELL_SIZE = 28 # Size to resize cell images for the model (e.g., 28x28 for MNIST-like models)
BORDER_CROP_RATIO = 0.15 # Crop 15% from each side of the cell to remove border noise

# --- Blank Cell Detection Parameters (Restored to previous best balance) ---
TOTAL_WHITE_PIXELS_THRESHOLD = 35 # Restored
DIGIT_ROI_WHITE_PIXEL_COUNT_THRESHOLD = 75 # Restored

# --- Contour Filtering Parameters (These were generally stable) ---
MIN_DIGIT_PIXEL_AREA_RATIO = 0.005 
MAX_DIGIT_PIXEL_AREA_RATIO = 0.60 
MIN_DIGIT_HEIGHT_RATIO = 0.30 
MAX_DIGIT_ASPECT_RATIO = 2.5 
MIN_DIGIT_ASPECT_RATIO = 0.15 


# Load the pre-trained Keras model
try:
    model = load_model(MODEL_PATH)
    print(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure '{MODEL_PATH}' is in the same directory as the script.")
    exit()

def preprocess_image(image_path):
    """Loads and preprocesses the sudoku image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
    
    # Apply adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C)
    
    return img, thresh

def find_sudoku_grid(thresh):
    """Finds the largest contour assumed to be the Sudoku grid."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000: 
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if area > max_area and len(approx) == 4:
                max_area = area
                largest_contour = approx 
            
    if largest_contour is None:
        print("Error: No large contour found (Sudoku grid).")
        return None
            
    approx = largest_contour.reshape((4, 2))
    
    s = approx.sum(axis=1)
    diff = np.diff(approx, axis=1)
    
    top_left = approx[np.argmin(s)]
    bottom_right = approx[np.argmax(s)]
    top_right = approx[np.argmin(diff)]
    bottom_left = approx[np.argmax(diff)]
    
    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")
    return corners


def extract_puzzle(img, corners):
    """Applies a perspective transform to extract the puzzle."""
    if corners is None:
        return None

    side = 450 
    dest = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(corners, dest)
    
    warped = cv2.warpPerspective(img, M, (side, side))
    
    return warped

def extract_digit_from_cell(cell_img, debug=False, cell_pos=None):
    """Extracts and preprocesses a digit from a single cell image."""
    
    gray_cell = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
    
    h_cell, w_cell = gray_cell.shape
    
    # Apply initial aggressive border cropping to remove any remaining grid lines
    border_crop = int(min(h_cell, w_cell) * BORDER_CROP_RATIO) 
    cropped_gray = gray_cell[border_crop:h_cell-border_crop, border_crop:w_cell-border_crop]

    if cropped_gray.shape[0] < 5 or cropped_gray.shape[1] < 5:
        if debug and cell_pos:
            cv2.imwrite(f"debug_cells/cell_{cell_pos}_cropped_too_small.png", np.zeros((CELL_SIZE,CELL_SIZE), dtype=np.uint8))
        return None
    
    # Binarize the cell using Otsu's thresholding and ensure white digit on black background
    if np.mean(cropped_gray) > 128: 
        cropped_gray = cv2.bitwise_not(cropped_gray)
    
    _, thresh = cv2.threshold(cropped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological opening to remove small noise
    kernel = np.ones(MORPH_OPEN_KERNEL,np.uint8) 
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=MORPH_OPEN_ITERATIONS)

    # 1. Blank Cell Detection (Initial quick check before contour finding)
    total_white_pixels = np.sum(thresh == 255)
    if total_white_pixels < TOTAL_WHITE_PIXELS_THRESHOLD: 
        if debug and cell_pos:
            cv2.imwrite(f"debug_cells/cell_{cell_pos}_blank_detected_total_px{total_white_pixels}.png", thresh)
        return None

    # 2. Find contours within the cell
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        if debug and cell_pos:
            cv2.imwrite(f"debug_cells/cell_{cell_pos}_no_contours.png", thresh)
        return None

    candidate_digit_roi = None
    max_candidate_area = 0

    cell_area_for_filters = thresh.shape[0] * thresh.shape[1] 
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        current_area = w * h
        aspect_ratio = float(w) / h
        height_ratio = float(h) / thresh.shape[0] 

        # 3. Contour Filtering (to distinguish digits from noise/lines)
        if (MIN_DIGIT_PIXEL_AREA_RATIO * cell_area_for_filters < current_area < MAX_DIGIT_PIXEL_AREA_RATIO * cell_area_for_filters and
            MIN_DIGIT_ASPECT_RATIO < aspect_ratio < MAX_DIGIT_ASPECT_RATIO and
            MIN_DIGIT_HEIGHT_RATIO < height_ratio):
            if current_area > max_candidate_area: 
                max_candidate_area = current_area
                candidate_digit_roi = thresh[y:y+h, x:x+w]
    
    if candidate_digit_roi is None:
        if debug and cell_pos:
            cv2.imwrite(f"debug_cells/cell_{cell_pos}_no_suitable_contour_final_filters.png", thresh)
        return None 

    # 4. Final White Pixel Check on the Extracted ROI
    digit_roi_white_pixel_count = np.sum(candidate_digit_roi == 255)
    if digit_roi_white_pixel_count < DIGIT_ROI_WHITE_PIXEL_COUNT_THRESHOLD: 
        if debug and cell_pos:
            cv2.imwrite(f"debug_cells/cell_{cell_pos}_low_roi_pixels_blank_detected_px{digit_roi_white_pixel_count}.png", candidate_digit_roi)
        return None

    # 5. Prepare for ML Model 
    h_digit, w_digit = candidate_digit_roi.shape
    
    if h_digit < 8 or w_digit < 8: 
        if debug and cell_pos:
            cv2.imwrite(f"debug_cells/cell_{cell_pos}_extracted_roi_too_small_final.png", candidate_digit_roi)
        return None

    # Pad to make it square if not already, using black borders
    if h_digit > w_digit:
        pad_size = (h_digit - w_digit) // 2
        temp_digit = cv2.copyMakeBorder(candidate_digit_roi, 0, 0, pad_size, h_digit - w_digit - pad_size, cv2.BORDER_CONSTANT, value=0)
    elif w_digit > h_digit:
        pad_size = (w_digit - h_digit) // 2
        temp_digit = cv2.copyMakeBorder(candidate_digit_roi, pad_size, w_digit - h_digit - pad_size, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        temp_digit = candidate_digit_roi

    # Resize to 20x20 and then pad to 28x28
    digit_resized = cv2.resize(temp_digit, (20, 20), interpolation=cv2.INTER_AREA)
    
    padded_digit = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    padded_digit[y_offset:y_offset+20, x_offset:x_offset+20] = digit_resized

    # Normalize pixel values to 0-1 and add batch dimension
    digit_for_model = padded_digit.astype('float32') / 255.0
    digit_for_model = np.expand_dims(digit_for_model, axis=-1) 
    digit_for_model = np.expand_dims(digit_for_model, axis=0) 

    if debug and cell_pos:
        debug_img_scaled = (digit_for_model.squeeze() * 255).astype(np.uint8)
        cv2.imwrite(f"debug_cells/cell_{cell_pos}_processed_for_ml.png", debug_img_scaled)

    return digit_for_model

def predict_digit(digit_img, confidence_threshold=0.75, return_confidence=False):
    """Predicts the digit using the loaded model."""
    predictions = model.predict(digit_img, verbose=0)
    digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    if return_confidence:
        return digit, confidence
    
    if confidence > confidence_threshold:
        return digit
    else:
        return 0 

def solve_sudoku(board):
    """Solves the Sudoku puzzle using backtracking."""
    
    def find_empty(bo):
        for r in range(9):
            for c in range(9):
                if bo[r][c] == 0:
                    return (r, c)
        return None

    def is_valid(bo, num, pos):
        # Check row
        for c in range(9):
            if bo[pos[0]][c] == num and pos[1] != c:
                return False
        # Check column
        for r in range(9):
            if bo[r][pos[1]] == num and pos[0] != r:
                return False
        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for r in range(box_y * 3, box_y * 3 + 3):
            for c in range(box_x * 3, box_x * 3 + 3):
                if bo[r][c] == num and (r, c) != pos:
                    return False
        return True

    find = find_empty(board)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1, 10):
        if is_valid(board, i, (row, col)):
            board[row][col] = i

            if solve_sudoku(board):
                return True

            board[row][col] = 0 # Backtrack

    return False

def recognize_sudoku_board(image_path):
    """
    Main function to recognize the Sudoku board from an image.
    """
    img, thresh = preprocess_image(image_path)
    if img is None:
        return None

    corners = find_sudoku_grid(thresh)
    if corners is None:
        print("Error: Could not find Sudoku grid in the image.")
        return None

    warped_puzzle_img = extract_puzzle(img, corners)
    if warped_puzzle_img is None:
        return None

    # Prepare for digit recognition
    grid = np.zeros((9, 9), dtype=int)
    
    # Create debug_cells directory and clear previous files
    debug_dir = "debug_cells"
    if DEBUG_MODE:
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)
        print(f"Debug images will be saved to '{debug_dir}/'")

    print("--- Recognizing Digits ---")
    for i in range(9):
        for j in range(9):
            cell_x = j * (warped_puzzle_img.shape[1] // 9)
            cell_y = i * (warped_puzzle_img.shape[0] // 9)
            cell_w = warped_puzzle_img.shape[1] // 9
            cell_h = warped_puzzle_img.shape[0] // 9

            cell_img = warped_puzzle_img[cell_y:cell_y + cell_h, cell_x:cell_x + cell_w]

            digit_img_for_prediction = extract_digit_from_cell(cell_img.copy(), debug=DEBUG_MODE, cell_pos=f"{i}_{j}")

            if digit_img_for_prediction is not None:
                digit, conf = predict_digit(digit_img_for_prediction, confidence_threshold=CONFIDENCE_THRESHOLD, return_confidence=True)
                
                if DEBUG_MODE:
                    debug_path_with_pred = f"{debug_dir}/cell_{i}_{j}_pred_{digit}_conf_{conf:.2f}.png"
                    debug_img_scaled = (digit_img_for_prediction.squeeze() * 255).astype(np.uint8)
                    cv2.imwrite(debug_path_with_pred, debug_img_scaled)

                if conf < CONFIDENCE_THRESHOLD:
                    print(f"Cell [{i},{j}] rejected: predicted {digit} with confidence={conf:.2f} (below threshold)")
                    grid[i, j] = 0
                else:
                    grid[i, j] = digit
                    print(f"Cell [{i},{j}] predicted {digit} with confidence={conf:.2f}")
            else:
                print(f"Cell [{i},{j}] is empty or unreadable (blank detection or no suitable contour).")
                grid[i, j] = 0

    return grid

if __name__ == "__main__":
    # --- Example Usage ---
    # Set this to one of your test images to confirm the restoration
    image_path = "images/sudoku3.jpg" # Example from earlier conversation
    # image_path = "images/image_332621.png" # The problematic image from current discussion
    # image_path = "images/image_cae2fb.png" # One of your other problematic boards
    # image_path = "images/image_a36232.png" # Example of another challenging board

    print(f"Processing image: {image_path}")
    predicted_board = recognize_sudoku_board(image_path)

    if predicted_board is not None:
        print("\n🔢 Predicted Sudoku Board (from image):")
        for row in predicted_board:
            print(row.tolist())
        
        board_to_solve = predicted_board.copy() 

        print("\nSolving Sudoku...")
        if solve_sudoku(board_to_solve):
            print("\n✅ Sudoku Solved Successfully!")
            print("\n🔢 Solved Sudoku Board:")
            for row in board_to_solve:
                print(row.tolist())
        else:
            print("\n❌ Could not solve the Sudoku board. This might be due to incorrect initial digit recognition or an unsolvable puzzle.")
    else:
        print("\nFailed to recognize Sudoku board from the image.")
# 🔢 AI-Powered Sudoku Solver from Image

This project is an end-to-end application that leverages computer vision and deep learning to automatically solve Sudoku puzzles from uploaded images. It provides a complete solution pipeline, from image input and digit recognition to puzzle solving and a user-friendly web interface.

## 🧠 How It Works

The application follows a modular pipeline to process and solve the Sudoku puzzle:

1.  **Image Preprocessing & Grid Extraction:** When an image is uploaded, it undergoes a series of OpenCV operations to prepare it for analysis.
    - The image is converted to grayscale to simplify pixel data.
    - Gaussian blurring is applied to reduce noise and enhance contours.
    - Adaptive thresholding is used to create a clear binary image, effectively separating the grid lines and digits from the background.
    - The largest quadrilateral contour is identified, which is assumed to be the Sudoku grid.
    - A perspective transform is applied to "un-warp" the grid, turning it into a perfectly square image for consistent processing.

2.  **Digit Recognition with a CNN:** Each of the 81 cells of the straightened grid is isolated and passed through a digit recognition pipeline.
    - A series of image filters (morphological operations, pixel-count checks) are applied to each cell to remove noise and isolate the primary digit contour.
    - The extracted digit is then resized and padded to a standard 28x28 pixel format, matching the input size of the deep learning model.
    - A **Convolutional Neural Network (CNN)**, trained on a combination of EMNIST and synthetically generated data, predicts the digit in the cell. A confidence threshold is used to filter out ambiguous or empty cells, marking them as `0`.

3.  **Sudoku Solving Algorithm:** The recognized 9x9 board is passed to a solver function.
    - A classic **backtracking algorithm** is implemented to recursively explore possible solutions. It checks for validity at each step, ensuring no numbers are repeated in the same row, column, or 3x3 subgrid.
    - The algorithm efficiently finds and returns the unique solution for the puzzle.

4.  **Interactive Streamlit UI:** All of these backend components are integrated into a clean web application.
    - The UI, built with **Streamlit**, handles image uploads and displays the processed results.
    - It presents the predicted board in an editable grid, allowing users to manually correct any misclassified digits before running the solver.
    - Finally, it displays the solved Sudoku puzzle in a clear table format.

## 🛠️ Technical Stack

- **Python:** The core programming language.
- **OpenCV:** For all computer vision and image manipulation tasks.
- **TensorFlow/Keras:** The deep learning framework used for the CNN.
- **NumPy:** For efficient numerical operations, especially with image data arrays.
- **Streamlit:** For building the interactive web application.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Bhuvan-08/Sudoku-Solver](https://github.com/Bhuvan-08/Sudoku-Solver)
    cd Sudoku-Solver
    ```

2.  **Set up the environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The project's dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    Start the application from your terminal.
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser, allowing you to start using the solver.

## 📁 File Structure

- `app.py`: The main Streamlit application file that orchestrates the entire process and handles the user interface.
- `sudoku_solver.py`: Contains the core Python logic, including the OpenCV pipeline, the digit classification function, and the backtracking solver.
- `improved_digit_cnn.h5`: The serialized Keras model file containing the weights of the trained CNN.
- `images/`: A directory for storing example Sudoku puzzle images for testing.
- `requirements.txt`: A file listing all the necessary Python packages for the project.

***

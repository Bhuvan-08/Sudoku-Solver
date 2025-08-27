import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Add the directory containing sudoku_solver.py to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now you can import from sudoku_solver
from sudoku_solver import recognize_sudoku_board, solve_sudoku, model

st.set_page_config(page_title="Sudoku Solver", layout="centered")

st.title("🔢 Sudoku Solver ")
st.markdown("Upload a Sudoku puzzle image, and let the magic happen!")

# --- Guidance for User Image Upload ---
st.warning("⚠️ Please upload a clear image where all numbers are distinctly visible. Images with watermarks, heavy shadows, or extreme angles may lead to incorrect recognition.")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a Sudoku image...", type=["png", "jpg", "jpeg"])

# Ensure the 'debug_cells' directory exists (it will be cleared by sudoku_solver.py on each run if DEBUG_MODE is True)
if not os.path.exists("debug_cells"):
    os.makedirs("debug_cells")

if uploaded_file is not None:
    # Save the uploaded file temporarily to process it with OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1) # Read as color image

    # Create a temporary path to save the image
    temp_image_path = "temp_sudoku_image.png"
    cv2.imwrite(temp_image_path, opencv_image)

    st.image(opencv_image, caption="Uploaded Sudoku Image", use_column_width=True)
    st.write("")
    st.info("Recognizing digits and solving the puzzle... this might take a moment.")

    try:
        # Step 1: Recognize the board
        st.subheader("Recognizing the Board...")
        recognized_board = recognize_sudoku_board(temp_image_path)

        if recognized_board is not None:
            st.success("Board recognition complete!")
            st.write("### Predicted Sudoku Board (Please correct any errors):")

            # Store the recognized board in session state for editing
            # Use a unique ID for the uploaded file to detect if a new file has been uploaded
            if 'last_uploaded_file_id' not in st.session_state or st.session_state.last_uploaded_file_id != uploaded_file.id:
                st.session_state.editable_board = recognized_board.tolist() # Convert numpy array to list for easier Streamlit handling
                st.session_state.last_uploaded_file_id = uploaded_file.id # Update the ID for the current upload
            
            edited_board = []
            
            # Apply custom CSS to reduce input field width and better align
            st.markdown(
                """
                <style>
                /* Target the div containing the number input to control its width */
                div.stNumberInput {
                    width: 50px !important; /* Adjust width as needed */
                }
                /* Target the columns to reduce padding and ensure even distribution */
                div[data-testid="column"] {
                    padding-left: 1px;
                    padding-right: 1px;
                    width: 10% !important; /* Adjust column width for grid-like appearance */
                    flex: 1 1 0%; /* Ensure columns flex equally */
                }
                /* Optional: Reduce vertical gap between elements in a block, specifically for input groups */
                div[data-testid="stVerticalBlock"] {
                    gap: 0rem; 
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            
            # Create an editable grid using number_input
            for r_idx in range(9):
                cols = st.columns(9) # Create 9 columns for each row to ensure side-by-side display
                row_values = []
                for c_idx in range(9):
                    # Use st.session_state.editable_board for initial value
                    current_value = st.session_state.editable_board[r_idx][c_idx]
                    
                    # Store whether the digit was initially recognized (to disable editing for "fixed" cells)
                    is_initial_digit = (recognized_board[r_idx][c_idx] != 0)
                    
                    with cols[c_idx]:
                        val = st.number_input(
                            label=f"R{r_idx}C{c_idx}", # Label is still needed but hidden
                            min_value=0, # Allow 0 for blank
                            max_value=9, # Allow digits 1-9
                            value=int(current_value), # Ensure integer value
                            key=f"cell_{uploaded_file.id}_{r_idx}_{c_idx}", # Unique key for each input, incorporates file ID for resetting
                            label_visibility="collapsed", # Hide the label to make the grid compact
                            disabled=is_initial_digit # Disable if it was an original recognized digit
                        )
                        row_values.append(val)
                edited_board.append(row_values)
            
            # Convert the list of lists back to a numpy array for solving
            edited_board_np = np.array(edited_board, dtype=int)

            # Button to trigger solving after user corrections
            if st.button("Solve Corrected Board"):
                st.subheader("Solving the Puzzle...")
                solvable_board = np.copy(edited_board_np) # Make a copy for solving
                if solve_sudoku(solvable_board):
                    st.success("Sudoku solved successfully!")
                    st.write("### Solved Sudoku Board:")
                    st.table(solvable_board)
                else:
                    st.error("❌ Could not solve the Sudoku board. This might be due to incorrect initial digit recognition (even after corrections) or an unsolvable puzzle.")
        else:
            st.error("Failed to recognize Sudoku board from the image. Please ensure it's a clear image of a Sudoku puzzle.")

    except Exception as e:
        st.exception(f"An error occurred during processing: {e}")
    finally:
        # Clean up the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
            
else:
    st.info("Please upload a Sudoku image to get started!")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit, OpenCV, and TensorFlow/Keras.")
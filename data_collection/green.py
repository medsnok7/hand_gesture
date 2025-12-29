import cv2
import numpy as np

# Function to apply chroma key (green screen removal)
def apply_chroma_key(green_frame, background_frame, lower_green, upper_green):
    """
    This function applies chroma keying to overlay the green_frame onto the background_frame.
    Any pixels in the green_frame that fall within the green color range (defined by lower_green
    and upper_green) will be replaced by the corresponding pixels from the background_frame.
    """
    # Ensure that the input frames are the same size
    if green_frame.shape[:2] != background_frame.shape[:2]:
        green_frame = cv2.resize(green_frame, (background_frame.shape[1], background_frame.shape[0]))

    # Create the mask for green colors
    green_mask = cv2.inRange(green_frame, lower_green, upper_green)

    # Ensure the mask is of type np.uint8
    if green_mask.dtype != np.uint8:
        green_mask = green_mask.astype(np.uint8)

    # Invert the mask so the green areas are set to 0
    mask_inv = cv2.bitwise_not(green_mask)

    # Extract the foreground (non-green parts of the green_frame)
    foreground = cv2.bitwise_and(green_frame, green_frame, mask=mask_inv)

    # Extract the background where the green parts are (from background_frame)
    background = cv2.bitwise_and(background_frame, background_frame, mask=green_mask)

    # Combine foreground and background
    combined = cv2.add(foreground, background)

    return combined

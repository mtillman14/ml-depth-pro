import cv2
import numpy as np

# Load the image
image_path = './data/blue_pool_table.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# Convert to HSV color space for better color segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color range for pool table cloth (adjust these based on the actual color)
lower_color = np.array([90, 50, 50])  # Lower bound of blue hue
upper_color = np.array([130, 255, 255])  # Upper bound of blue hue

# Create a mask to isolate the cloth
cloth_mask = cv2.inRange(hsv, lower_color, upper_color)

# Optionally, apply morphological operations to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, kernel)

# Use the mask to isolate the cloth region in the image
cloth_region = cv2.bitwise_and(image, image, mask=cloth_mask)

# Convert the masked cloth region to grayscale for edge detection
gray_cloth = cv2.cvtColor(cloth_region, cv2.COLOR_BGR2GRAY)

# Perform edge detection on the cloth region
edges = cv2.Canny(gray_cloth, 50, 150)

# Find contours based on edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour which should correspond to the cloth's boundary
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to reduce complexity and focus on the general shape
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Draw the approximated contour on the original image
    cv2.drawContours(image, [approx_contour], -1, (0, 255, 0), 3)
else:
    print("No contour found for the cloth boundary.")

# Display the result
cv2.imshow("Cloth Boundary Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

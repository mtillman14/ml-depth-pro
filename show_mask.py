import cv2
import numpy as np

# Load the image
image_path = './data/pool_balls.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# Display the image and let the user select the ROI
cv2.imshow("Select the cloth area", image)
roi = cv2.selectROI("Select the cloth area", image, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("Select the cloth area")

# Extract the selected region of interest (ROI)
x, y, w, h = roi
selected_region = image[y:y+h, x:x+w]

# Convert the selected region to HSV
hsv_region = cv2.cvtColor(selected_region, cv2.COLOR_BGR2HSV)

# Calculate the average color of the selected region in HSV
avg_color = hsv_region.mean(axis=0).mean(axis=0)
print(f"Average HSV color: {avg_color}")

# Define a color range based on the average color
# You may want to adjust the ranges below to allow for variations
hue = int(avg_color[0])
saturation = int(avg_color[1])
value = int(avg_color[2])

lower_bound = np.array([hue - 10, saturation - 40, value - 40])
upper_bound = np.array([hue + 10, saturation + 40, value + 40])

print(f"Lower HSV bound: {lower_bound}")
print(f"Upper HSV bound: {upper_bound}")

# Create a mask based on the calculated color range
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Display the mask
cv2.imshow("Mask of the selected color range", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2 

# Define image dimensions for resizing
dim = (300, 300)


for i in range(10):
    # Construct full image paths using correct directory separator
        img1 = cv2.imread(f"data/cat.{i}.jpg", cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(f"data/dog.{i}.jpg", cv2.COLOR_BGR2GRAY)

        # Resize images with appropriate interpolation for grayscale
        resized1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
        resized2 = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

        # Write resized images to output directory
        cv2.imwrite(f"data2/cat.{i}.jpg", resized1)
        cv2.imwrite(f"data2/dog.{i}.jpg", resized2)
# Close any open windows (optional)
cv2.destroyAllWindows()

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt
import os


# Load and preprocess the image
def load_and_preprocess_image(image_path):
    """Load and preprocess the image"""
    try:
        # Check if the file exists
        if not os.path.exists(image_path):
            print(f"Error: File does not exist - {image_path}")
            # Try to find similar files
            dir_path = os.path.dirname(image_path)
            file_name = os.path.basename(image_path)
            if os.path.exists(dir_path):
                print(f"Contents of the directory: {os.listdir(dir_path)}")
            return None

        # Check if the file is an image
        try:
            img = Image.open(image_path)
            img.verify()  # Verify the integrity of the image file
            img = Image.open(image_path)  # Reopen the image as the file pointer may be reset after verify()
        except Exception as e:
            print(f"Error: Unable to open the image - {e}")
            return None

        # Get the width and height of the image
        width, height = img.size
        # Convert to a numpy array
        image_array = np.array(img)
        print(f"Original image shape: {image_array.shape}")

        # Ensure the image is in RGB format
        if image_array.shape[2] == 4:  # If it is in RGBA format
            image_array = image_array[:, :, :3]  # Remove the Alpha channel
            print("Converted from RGBA format to RGB format")

        # Additional verification: Try to display image size information
        print(f"Image size: {width} x {height} pixels")
        print(f"Image mode: {img.mode}")

        return image_array
    except Exception as e:
        print(f"Error loading the image: {e}")
        return None


# Define multiscale kernels
def get_multiscale_kernels():
    """Get multiscale kernels"""
    # Small-scale edge detection kernels
    small_kernels = [
        np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),  # Vertical edge
        np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),  # Horizontal edge
        np.array([[1, 1, 1], [0, 1, 0], [-1, 2, -1]]),  # Diagonal edge
    ]

    # Medium-scale feature extraction kernels
    medium_kernels = [
        np.array([[1, 1, 1], [1, 5, 2], [-1, -1, -1]]),
        np.array([[1, -3, 2], [1, -2, 1], [1, -1, 2]]),
        np.array([[-1, 3, 2], [1, -2, 1], [1, 0, -2]]),
    ]

    # Large-scale feature extraction kernels
    large_kernels = [
        np.array([[1, 0, -5, 0, 2], [0, -3, 0, 1, 0], [1, 0, -2, 0, 1], [0, 1, 0, -2, 0], [1, 0, -2, 0, -1]]),
        np.array([[1, 0, 0, 7, 0, 1, -1],
                  [1, -2, 0, 10, 4, -1, 1],
                  [2, 4, 0, 2, -1, 0, 1],
                  [3, -2, 5, 0, 1, -3, 1],
                  [2, -1, 0, -1, 0, 3, 1],
                  [2, -1, 6, 1, 1, -1, 1],
                  [1, 0, 8, -1, 3, -1, 1]]),
    ]

    return small_kernels, medium_kernels, large_kernels


# Extract features using kernels
def extract_features(image, kernels):
    """Extract image features using given kernels"""
    # Store convolution results
    feature_maps = []

    # Perform convolution for each channel of the image
    for kernel in kernels:
        convolved_channels = []
        for channel in range(image.shape[2]):
            channel_data = image[:, :, channel]
            # Apply the kernel, use'same' mode to keep the output size the same as the input
            convolved_channel = convolve2d(channel_data, kernel, mode='same', boundary='symm')
            convolved_channels.append(convolved_channel)

        # Recombine the convolved channels into a feature map
        feature_map = np.stack(convolved_channels, axis=2)
        feature_maps.append(feature_map)

    # Concatenate all feature maps along the channel dimension
    if feature_maps:
        concatenated_features = np.concatenate(feature_maps, axis=2)
        return concatenated_features
    else:
        return None


# Main function
def main():
    # Set the file path
    image_path = r"E:\F盘的东西9.13\脑瘤高光谱数据\12.26实验数据\rgb\1.jpg"
    output_path = r".\28_features.mat"

    # Print path information for debugging
    print(f"Attempting to load the image: {image_path}")
    print(f"Saving features to: {output_path}")

    # Verify if the directory exists
    image_dir = os.path.dirname(image_path)
    if not os.path.exists(image_dir):
        print(f"Error: The image directory does not exist - {image_dir}")
        return

    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)
    if image is None:
        print("Unable to load the image, the program will exit")
        return

    # Get multiscale kernels
    small_kernels, medium_kernels, large_kernels = get_multiscale_kernels()

    # Extract features at different scales
    print("Extracting small-scale features...")
    small_scale_features = extract_features(image, small_kernels)

    print("Extracting medium-scale features...")
    medium_scale_features = extract_features(image, medium_kernels)

    print("Extracting large-scale features...")
    large_scale_features = extract_features(image, large_kernels)

    # Concatenate all features and the original RGB image along the channel dimension
    print("Concatenating features...")

    # Create a new 3-channel multiscale feature map
    multiscale_feature_map = np.zeros_like(image, dtype=np.float32)

    # Merge features of each scale into 3 channels
    scale_weights = [0.3, 0.4, 0.3]  # Assign weights to different scales
    scale_features = [small_scale_features, medium_scale_features, large_scale_features]

    for i, (features, weight) in enumerate(zip(scale_features, scale_weights)):
        if features is not None:
            # Calculate the average of the current scale features as a single-channel representation
            channel_avg = np.mean(features, axis=2, keepdims=True)
            # Add the current scale features to the corresponding channel of the multiscale feature map with weight
            channel_idx = i % 3  # Cycle through 3 channels
            multiscale_feature_map[:, :, channel_idx] += weight * channel_avg.squeeze()

    # Concatenate the original RGB image and the multiscale feature map
    final_feature_map = np.concatenate([image, multiscale_feature_map], axis=2)
    print(f"Final feature map shape: {final_feature_map.shape}")

    # Save as a MAT file
    print(f"Saving to {output_path}...")
    try:
        sio.savemat(output_path, {'features': final_feature_map})
        print("Save successful!")
    except Exception as e:
        print(f"Error saving the file: {e}")

    # Optional: Display the original image and the multiscale feature map
    plt.figure(figsize=(15, 10))

    plt.subplot(1, 2, 1)
    plt.title('Original RGB Image')
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Multiscale Feature Map')
    # Display the brightness of the multiscale feature map (average of all channels)
    plt.imshow(np.mean(multiscale_feature_map, axis=2), cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('feature_visualization.png')
    plt.show()


if __name__ == "__main__":
    main()
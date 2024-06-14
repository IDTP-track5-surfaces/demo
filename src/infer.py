import argparse
import tensorflow as tf
import datetime
import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
from model import create_model
import matplotlib.pyplot as plt

def min_depth_over_time_plot(prediction):
    """
    Plot the maximum depth over time.
    """
    min_depth = np.min(prediction[:,:,:,0], axis=(1, 2))
    plt.figure(figsize=(12, 6))
    plt.plot(min_depth)
    plt.title('Minimum Depth Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Depth')
    plt.grid()
    plt.tight_layout()
    plt.savefig('logs/min_depth_over_time.png')
    plt.close()



def plot_inference(infer_refracted, infer_reference, infer_predictions):
    num_samples = infer_refracted.shape[0]
    num_pages = (num_samples + 2) // 3  # Calculate how many pages are needed

    for page_idx in range(num_pages):
        plt.figure(figsize=(12, 12))  # Create a new figure for each page
        for i in range(3):
            idx = page_idx * 3 + i
            if idx >= num_samples:
                break  # Stop if the index goes beyond the number of samples

            ax1 = plt.subplot(3, 4, 4*i + 1)
            ax1.imshow(infer_refracted[idx].numpy())
            ax1.set_title('Refracted Image')
            ax1.axis('off')

            ax2 = plt.subplot(3, 4, 4*i + 2)
            ax2.imshow(infer_reference[idx].numpy())
            ax2.set_title('Reference Image')
            ax2.axis('off')

            ax3 = plt.subplot(3, 4, 4*i + 3)
            im_pred = ax3.imshow(infer_predictions[idx, :, :, 0], cmap='viridis')
            ax3.set_title('Predicted Depth')
            ax3.axis('off')

            plt.colorbar(im_pred, ax=ax3, fraction=0.046, pad=0.04)

        plt.tight_layout()
        page_path = f"logs/inference_page_{page_idx+1}.png"
        os.makedirs(os.path.dirname(page_path), exist_ok=True)  # Ensure the directory exists
        plt.savefig(page_path)
        plt.close()  # Close the plot to free memory

def load_image_as_tensor(image_path, image_size=(128, 128)):
    """
    Load an image file as a TensorFlow tensor in grayscale and resize it.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3) # Adjust channels for grayscale
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image

def load_tensors(data_path):
    refracted_files = [os.path.join(root, f) for root, _, files in os.walk(data_path) for f in files if f.endswith('.jpg')]
    refracted_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(refracted_files)
    print(f"Found {len(refracted_files)} refracted images.")
    for file in refracted_files:
        print(file)

    refracted_tensors_list = []
    reference_tensors_list = []
    for file in refracted_files:
        refracted_tensor = load_image_as_tensor(file)
        refracted_tensors_list.append(refracted_tensor)
        reference_tensors_list.append(refracted_tensors_list[0])
    
    input_tensors = tf.concat([refracted_tensors_list, reference_tensors_list], axis=-1)

    return input_tensors

def infer(data_path, model_path): 
    model, custom_objects = create_model()

    input_tensors = load_tensors(data_path)
    print("Input tensors shape: ", input_tensors.shape)

    loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    print("Predicting...")
    predictions = loaded_model.predict(input_tensors)
    print("Predictions shape: ", predictions.shape)

    plot_inference(input_tensors[:, :, :, :3], input_tensors[:, :, :, 3:], predictions)
    min_depth_over_time_plot(predictions)



if __name__ == "__main__":
    data_path = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/demo/frames/11-22-55"
    model_path = "/Users/mohamedgamil/Desktop/Eindhoven/block3/idp/code/demo/model/final_model.h5"
    infer(data_path, model_path)

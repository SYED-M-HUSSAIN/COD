import os
import random
import shutil

# Define your dataset directory and subdirectories
base_dir = 'dataset'  # Change this to your dataset directory
image_dir = os.path.join(base_dir, 'images')  # Change to your image directory
label_dir = os.path.join(base_dir, 'labels')  # Change to your label directory

# Create the train and validation directories if they don't exist
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# List the files in your image and label directories
image_files = os.listdir(image_dir)
label_files = os.listdir(label_dir)

# Shuffle the lists of image and label files to ensure a random distribution
random.shuffle(image_files)
random.shuffle(label_files)

# Calculate the number of samples for the training and validation sets based on the desired ratio
total_samples = len(image_files)
train_ratio = 0.85
val_ratio = 0.15
num_train_samples = int(total_samples * train_ratio)
num_val_samples = total_samples - num_train_samples

# Copy the images and labels to the training and validation directories
for i in range(num_train_samples):
    image_src = os.path.join(image_dir, image_files[i])
    label_src = os.path.join(label_dir, label_files[i])
    
    image_dst = os.path.join(train_dir, image_files[i])
    label_dst = os.path.join(train_dir, label_files[i])
    
    shutil.copy(image_src, image_dst)
    shutil.copy(label_src, label_dst)

for i in range(num_train_samples, total_samples):
    image_src = os.path.join(image_dir, image_files[i])
    label_src = os.path.join(label_dir, label_files[i])
    
    image_dst = os.path.join(val_dir, image_files[i])
    label_dst = os.path.join(val_dir, label_files[i])
    
    shutil.copy(image_src, image_dst)
    shutil.copy(label_src, label_dst)

print(f"Split {total_samples} samples into {num_train_samples} for training and {num_val_samples} for validation.")

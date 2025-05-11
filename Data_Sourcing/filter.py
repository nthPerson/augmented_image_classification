import shutil
import os
# This Code uses GPT
# Path to the filtered validation annotations file
filtered_file = 'filtered_val_annotations.txt'  # Adjust path as needed

# Path to the Tiny ImageNet 'val' directory
val_dir = 'images'  # Replace with the correct path

# Path to your custom validation folder
custom_val_dir = './custom_validation_data/'  # Adjust path as needed

# List of selected synset IDs (these are your 10 classes)
selected_synsets = [
    "n02099601", "n02123394", "n02129165", "n02132136", "n02403003",
    "n02415577", "n02423022", "n02481823", "n02504458", "n02509815"
]

# Create the directories for each class if they don't exist
for synset_id in selected_synsets:
    os.makedirs(os.path.join(custom_val_dir, synset_id), exist_ok=True)

# Read the filtered text file and copy the relevant images
with open(filtered_file, 'r') as infile:
    for line in infile:
        image_filename, synset_id, *_ = line.split()

        # Only copy the images that belong to your selected synset IDs
        if synset_id in selected_synsets:
            # Construct the source image path
            source_path = os.path.join(val_dir, image_filename)

            # Construct the destination image path
            dest_path = os.path.join(custom_val_dir, synset_id, image_filename)

            # Copy the image to the corresponding class folder
            shutil.copy(source_path, dest_path)

print("Image filtering and copying complete.")

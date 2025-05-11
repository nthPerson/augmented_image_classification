# List of your 10 selected synset IDs
#This Code Uses GPT
import os
print("Current working directory:", os.getcwd())
selected_synsets = [
    "n02099601", "n02123394", "n02129165", "n02132136", "n02403003",
    "n02415577", "n02423022", "n02481823", "n02504458", "n02509815"
]

# Path to the validation annotations text file
input_file = 'val_annotations.txt'  # Adjust this to the correct path in your project
output_file = 'filtered_val_annotations.txt'  # Path for the filtered list

# Open the input text file and the output file to write filtered annotations
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    # Read each line in the text file
    for line in infile:
        # Split the line into image filename and synset ID
        image_filename, synset_id, *_ = line.split()

        # If the synset ID is in the list of selected synsets, write to the output file
        if synset_id in selected_synsets:
            outfile.write(line)

print("Filtering complete. The filtered list is saved in", output_file)

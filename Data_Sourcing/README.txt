val.py: This script filters the original val_annotations.txt provided by the 200 TinyImageNet classes, so it only contains entries that belong to 10 specific synset IDs. 

Functionality: 
- defines the list of the 10 chosen synsets (chosen to be semantically related). 
- reads the complete val_annotations.txt file. 
- for each line in the val_annotations.txt that maps a certain synset id to a category, it checks if the given class exists in the list. If so, it creates a new text file called filtered_val_annotations.txt that will house the picked entries. 
- The result is a text file,  filtered_val_annotation.txt, containing  entries that only belong to the 10 specific synset IDs. 


filter.py: This script copies the images specified in the filtered_val_annotations.txt into a new directory structure. This structure is organized by each of the 10 subclasses.

Functionality:

- defines the  10 selected_synsets
- creates a folder for each of the classes inside the directory,custom_validation_data. 
- After reading filtered_val_annotations.txt that was outputted from val.py, it checks each syset ID is in the list and copies the image from the original directory into the corresponding class folder. 

Overall, the filtered data contained the following subclasses:

Synset 	  category
n02099601 golden retriever
n02123394 Persian cat
n02129165 lion, king of beasts, Panthera leo
n02132136 brown bear, bruin, Ursus arctos
n02403003 ox
n02415577 bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain
          sheep, Ovis canadensis
n02423022 gazelle
n02481823 chimpanzee, chimp, Pan troglodytes
n02504458 African elephant, Loxodonta Africana
n02509815 lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens
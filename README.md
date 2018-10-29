# Predict Tags by Learning Recipe Embeddings  

This is capstone project repository collaborated with Plated. Our goal is to help Plated to build a deep learning model to auto-generate recipe tags using both cooking instruction, dish images, and other data sources. We utilized mutiple deep learning models to obtain comprehensive recipe representations, then use the representation to predict multiple tags. 

## Model
* Recipe1M-instruction data: A Skip-Gram model to learn recipe language embeddings 
* Cooking instruction data: A Two-Stage LSTM to obtain intruction embedding using self-pretrained recipe language embeddings
* Dish Image: A deep neural network (Resnet 50) to learn recipe image representation

## Data
* [Recipe1M](http://im2recipe.csail.mit.edu/dataset/download/)
* [Cooking Insturctions](/data/data_sample.pdf)
* [Dish Images](/data/data_sample.pdf)
* [Tags](/data/Tags_structure_self.csv): includes cuicine, cooking difficulty levels, [more](/data/Tags_structure_self.csv)
## Related Project 
[Learning Cross-modal Embeddings for Cooking Recipes and Food Images](http://pic2recipe.csail.mit.edu)

![Logo](pics/logo_image.png) 

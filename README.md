# Predict Tags by Learning Recipe Representation  

This is capstone project repository collaborated with Plated. Our goal is to help Plated build a deep learning model to auto-generate recipe tags using both cooking instruction, dish images, and other data sources. We successfully build models on top of instruction data with high AUC, where we applied single-task to make prediction of 9 cuisine tags individually and a multi-task module to obtain comprehensive recipe representations then use the it to predict multiple tags all together.

## Model
* Recipe1M-instruction data: A Skip-Gram model to learn recipe language embeddings 
* Cooking instruction data: A Two-Stage LSTM/GRU to obtain intruction representation using self-pretrained recipe language/GloVe embeddings in both single-tasking and multi-tasking manner
* Dish Image: A deep neural network (Resnet 50) to learn recipe image representation

## Data
* [Recipe1M](http://im2recipe.csail.mit.edu/dataset/download/)
* [Cooking Insturctions](/data/data_sample.pdf)
* [Dish Images](/data/data_sample.pdf)
* [Tags](/data/Tags_structure_self.csv): includes cuisine, cooking difficulty levels, [more](/data/Tags_structure_self.csv)

## Related Project 
[Learning Cross-modal Embeddings for Cooking Recipes and Food Images](http://pic2recipe.csail.mit.edu)

## 
Team Member (DataZoo): Tingyan Xiang, Hetian Bai, Jieyu Wang, Cong Liu

Industral Metor: Ph.D Andrew Marchese 

![Logo](pics/logo_image.png) 

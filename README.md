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
* [Tags](/data/Tags_structure_self.csv): includes cuisine, cooking difficulty levels, [more](/data/Tags_structure_self.csv)

## Work Process 
- [x] Data Preprocessing and Analysis
- [x] Model (baseline) on Intruction Data to predict crucines (binary and multi-tasking) 
- [ ] Train Recipe Language Embeddings
- [ ] Retrain and improve Intruction Model using pretrained embedding
- [ ] Image processing
- [ ] Build model for image representation 
- [ ] Multi-tasking model to predict multiple tags

## Memo
Our team collaborated with Plated. We have discussed our project in detail with the data science team from Plated, and Doctor Marchese meets with us weekly (biweekly  sometimes)  as our supervisor.  For our project, the main task is to predict tags for recipes and achieve reasonable embeddings at the same time. There are some classes for tags, such as cuisine style and difficulty level, and we have recipe data including text and pictures from Plated. With respect to evaluation, metrics, such as accuracy and AUC, will be used to measure results of our models. Besides, Plated may take advantage of hidden representation from our models to do recommendation and test the quality of hidden representation. 

So far, we have finished data collection, data clean and part of modeling work. We got all necessary data from Plated and preprocessed text data. With instruction data and cuisine tags, we have built binary classification models, and achieved above 80% AUC for most cuisine tags. Based on binary classification models, we have build a multi-task model for all cuisine tags, which outperforms the single binary classification model for some tags. In the following time, we plan to train ourself word embeddings in the recipe domain and use pictures data in our model. 

## Related Project 
[Learning Cross-modal Embeddings for Cooking Recipes and Food Images](http://pic2recipe.csail.mit.edu)

## 
Team Member (DataZoo): Tingyan Xiang, Hetian Bai, Jieyu Wang, Cong Liu

Industral Metor: Ph.D Andrew Marchese 

![Logo](pics/logo_image.png) 

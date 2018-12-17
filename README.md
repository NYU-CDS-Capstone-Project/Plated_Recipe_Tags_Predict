# Predict Tags by Learning Recipe Representation  

This is capstone project repository collaborated with Plated. Our goal is to help Plated build a deep learning model to auto-generate recipe tags using both cooking instruction, dish images, and other data sources. We successfully build models on top of instruction data with high AUC, where we applied single-task to make prediction of 9 cuisine tags individually and a multi-task module to obtain comprehensive recipe representations then use the it to predict multiple tags all together.

## Data
* [Recipe1M](http://im2recipe.csail.mit.edu/dataset/download/)
* [Cooking Insturctions](/data/data_sample.pdf)
* [Dish Images](/data/data_sample.pdf)
* [Tags](/data/Tags_structure_self.csv): includes cuisine, cooking difficulty levels, [more](/data/Tags_structure_self.csv)

## Model
* Recipe1M-instruction data: A Skip-Gram model to learn recipe language embeddings (Domain-Edmbd)
* Cooking instruction data: A Two-Stage LSTM/GRU to obtain intruction representation using self-pretrained recipe language/GloVe embeddings in both single-tasking and multi-tasking manner
* Dish Image: A deep neural network (Resnet 50) to learn recipe image representation

## Results
#### Instruction Model -- K-fold (mean) Validation AUC on Cuisine Tag Prediction under Variance Settings
| Cuisine Category| Tags Percentage	| GRU 	| GRU + Aug	| LSTM + Aug	| GRU + Domain-Edmbd + Aug|Multi-task |
| :--- 				|:---:			|:---:	|:---:	|:---:		|:---:		| ---: 	|
|American |27.35% |0.80612 |0.81249 |0.79381 |0.69369 |0.74103|
|Italian |23.33%| 0.88027 |0.91504 |0.89848| 0.80436 |0.85489|
|Asian |18.22% |0.97855 |0.97919 |0.97982 |0.88072 |0.94860|
|Latin-Ame |9.49% |0.90628 |0.94837 |0.85706 |0.92311 |0.93433|
|French |7.74% |0.74977 |0.80471 |0.77272 |0.85640 |0.79605|
| Mediterranean| 7.66% |0.73317 |0.75837 |0.72589 |0.75442 |0.79292|
|Mid-east |4.63% |0.81138| 0.81850 |0.78675 |0.77870| 0.87369|
|Indian |2.35% |0.78643 |0.87356 |0.73456 |0.87249 |0.88438|
|Mexican |1.36% |0.67503 |0.70365 | 0.73999 |0.74288 |0.90554|

#### Image Model -- K-fold (mean) Validation AUC on Cuisine Tag Prediction
| Cuisine Category| American | Italian| Asian|Latin-Ame|French|
| AUC| 0.7719| 0.8810| 0.7411| 0.7235| 0.7188|

#### Recipe Representation Visualization
![Recipe Embedding](https://github.com/NYU-CDS-Capstone Project/Plated_Recipe_Tags_Predict/blob/master/pics/Recipe%20Representation.png){:height="50%" width="50%"}

## Related Project 
[Learning Cross-modal Embeddings for Cooking Recipes and Food Images](http://pic2recipe.csail.mit.edu)

## 
Team Member (DataZoo): Tingyan Xiang, Hetian Bai, Jieyu Wang, Cong Liu

Industral Metor: Ph.D Andrew Marchese 

![Logo](pics/logo_image.png) 

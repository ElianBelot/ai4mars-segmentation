# Terrain classification for Mars rovers

# **Introduction**

This project focuses on the problem of terrain classification for Mars rovers. This task is essential for future autonomous rover missions, as it can help rovers navigate safely and efficiently on the Martian surface. We used a dataset consisting of 35K images from Curiosity, Opportunity, and Spirit rovers with semantic segmentation labels collected through crowdsourcing. The dataset also includes 1.5K test labels annotated by the rover planners and scientists from NASA's MSL (Mars Science Laboratory) and MER (Mars Exploration Rovers) missions.

The full code for this experiment is available [here](https://www.kaggle.com/code/raskotv/segmentation-with-resnet50/notebook?scriptVersionId=127484960) as a Kaggle notebook.

![The Mars Curiosity Rover [(source)](https://mars.nasa.gov/msl/home/).](assets/Untitled.png)

The Mars Curiosity Rover [(source)](https://mars.nasa.gov/msl/home/).

# **Dataset**

The dataset we used for training and validating the terrain classification model was created specifically for this purpose. It contains approximately 326K semantic segmentation full image labels on 35K images taken by the Curiosity, Opportunity, and Spirit rovers. Each image in the dataset was labeled by 10 different people to ensure high quality and agreement of the crowdsourced labels.

![An original MSL NAVCAM image along with a high rated AI4MARS label [(source)](https://data.nasa.gov/Space-Science/AI4MARS-A-Dataset-for-Terrain-Aware-Autonomous-Dri/cykx-2qix).](assets/Untitled%201.png)

An original MSL NAVCAM image along with a high rated AI4MARS label [(source)](https://data.nasa.gov/Space-Science/AI4MARS-A-Dataset-for-Terrain-Aware-Autonomous-Dri/cykx-2qix).

In addition to the crowdsourced labels, the dataset also includes around 1.5K validation labels annotated by the rover planners and scientists from NASA's MSL and MER missions, which operate the Curiosity, Spirit, and Opportunity rovers. These expert-annotated labels serve as a valuable resource for validating the performance of the trained model.

The segmentation is identified by the following classes:

```
0 - soil
1 - bedrock
2 - sand
3 - big rock
4 - no label
```

# **Model**

We utilized a deep learning-based approach for the terrain classification task. Specifically, we finetuned a U-Net architecture implemented using ResNet50. The model was trained using PyTorch Lightning, a high-level wrapper around PyTorch that simplifies the training process.

![The U-Net architecture used for semantic segmentation [(source)](https://www.frontiersin.org/articles/10.3389/fnagi.2022.841297/full).](assets/Untitled%202.png)

The U-Net architecture used for semantic segmentation [(source)](https://www.frontiersin.org/articles/10.3389/fnagi.2022.841297/full).

# Training

The training was carried out on Kaggle, using a P100 GPU to accelerate the computations.
We used a batch size of 32 and trained the model for 30 epochs on 15,000 images.
We used 20% of the dataset to create a validation set, and the rest of it for the training set.
Metrics were tracked using Weights and Biases.

The training set contains a cleaned and merged version of crowdsourced labels, with a minimum agreement of 3 labelers and 2/3 agreement for each pixel. The rover and distances further than 30 meters are masked.

### **Loss**

![Training loss per epoch](assets/Untitled%203.png)

Training loss per epoch

Training loss went down over time, which shows the model is converging.

![Validation loss per epoch](assets/Untitled%204.png)

Validation loss per epoch

However, validation loss initially decreased, but steadily increased during training.
Without further context, one might posit that this is due to overfitting.
But this is unlikely, since as we’ll see the model is performing better on relevant metrics even within the validation set. This small increase in loss on the validation set is more likely to be a consequence of the diversity and complexity of segmentation datasets like this one.

### **Accuracy**

In the context of segmentation models, accuracy is a metric used to measure how well a model correctly identifies the pixels in an image that belong to a particular class or category. It is defined as the ratio of the total number of correctly classified pixels to the total number of pixels in the image.

![Validation accuracy per epoch](assets/Untitled%205.png)

Validation accuracy per epoch

The accuracy went up during the first 20 epochs, before starting to plateau, eventually reaching around 87%.

### Intersection over Union

The IoU (Intersection over Union), also known as Jaccard index, is a metric used to measure the similarity between the predicted segmentation map $P$ and the ground truth segmentation map $G$ of an image. It is defined as follows:

$$
IoU = \frac{|P \cap G|}{|P \cup G|}

$$

IoU values range from 0 to 1, where a value of 1 indicates a perfect overlap between the predicted and ground truth segmentation maps, while a value of 0 indicates no overlap at all.

![Validation IoU per epoch](assets/Untitled%206.png)

Validation IoU per epoch

The IoU values are expectedly lower than the accuracy values, as it is a tighter metric.
Nevertheless, it steadily rose throughout training before starting to plateau at around 20 epochs in, reaching a value of around 68%.

# Evaluation

The model was evaluated on a test set of 322 expert-annotated images with ~1.5k segmentations.
These images were annotated by rover planners and scientists from NASA.

First, the following is a qualitative demonstration of the model’s ability on a few images.

![Original image, crowdsourced segmentation, and predicted segmentation by the model for 3 different images.](assets/Untitled%207.png)

Original image, crowdsourced segmentation, and predicted segmentation by the model for 3 different images.

We evaluate the model on accuracy and IoU, and also present a confusing matrix for 3 different test sets. Although they contain the same images, each test set differs in the degree of agreement of expert labelers needed to validate a segmentation.

The sparsest test set requires a minimum of 3 labelers to agree, and the two others require 2 agreements and 1 agreement respectively.

![Breakdown of proportions of how many expert labelers labeled a pixel a specific class for the MSL test set [(source)](https://www.frontiersin.org/articles/10.3389/fnagi.2022.841297/full).](assets/Untitled%208.png)

Breakdown of proportions of how many expert labelers labeled a pixel a specific class for the MSL test set [(source)](https://www.frontiersin.org/articles/10.3389/fnagi.2022.841297/full).

As we can see in the above figure, the “Big rock” segmentations are the most controversial one, as only a small portion of the labels achieves 3 agreements. We’ll see this reflected in the confusion matrices presented below.

As a key for the confusion matrices, here are the labels and their corresponding numerical values:

```
0 - soil
1 - bedrock
2 - sand
3 - big rock
4 - no label
```

### 1 agreement

```
Accuracy: 0.8451
IoU: 0.7360
```

![Untitled](assets/Untitled%209.png)

This is the densest test set as it only requires one labeler to agree on a given segmentation.
As a consequence, it’s easier for the model to achieve higher accuracy and IoU, and thus this test set is the one where the model achieves its best performance.

As we can see, this confusion matrix is the most spread out, as it is less concentrated in the diagonal compared to the following two. This is also explained by the low-threhshold for agreement leading to a lot more classes needing to be predicted.

### 2 agreements

```
Accuracy: 0.8180
IoU: 0.7029
```

![Untitled](assets/Untitled%2010.png)

This matrix is more concentrated one the diagonal, despite the model achieving lower accuracy overall. As expected due to their controversial labeling, big rocks (category 3) are the hardest for the model to classify correctly, and are often not labeled (category 4).

### 3 agreements

```
Accuracy: 0.7262
IoU: 0.5917
```

![Untitled](assets/Untitled%2011.png)

The same patterns as the previous test set emerge here. This is the most demanding test set in terms of accuracy, which is reflected in the performance of the model. The fact that the lowest row is denser is to be expected, since category 4 is a “no label” category and therefore groups a lot of different possible objects on the image.

Overall, this model achieves around 95% accuracy on soil, bedrock and sand.
The drop in overall accuracy is due to the misclassification of some classes that are not supposed to be labeled (again due to the sparse nature of this test set), and the fact that “Big rock” seems to be an ambiguous category, especially for humans.

# **Caveats and limitations**

There are some caveats and limitations to consider when interpreting the results of this project:

1. 1. The model was trained on a subset of the available data (15,000 images). Training on a larger dataset could potentially improve the model's performance.
2. The dataset consists of images from three different rovers, each with its own camera system and specifications. This may introduce some variability in the data, which the model might need to account for during training.
3. The semantic segmentation labels in the dataset were obtained through crowdsourcing, which could introduce some noise and inconsistencies in the labels. However, each image was labeled by 10 different people to minimize this issue.
4. We used a single model architecture (U-Net) for the terrain classification task. Exploring other architectures or using an ensemble of models could potentially lead to better performance.
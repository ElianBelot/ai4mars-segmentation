<!--- Banner -->
<br />
<p align="center">
  <a href="#"><img src="https://i.ibb.co/RDwV3cH/image.png"></a>
  <h3 align="center">Mars Terrain Classification</h3>
  <p align="center">Image segmentation on the AI4MARS dataset using ResNet50 and PyTorch Lightning.</p>
</p>

<!--- Introduction --><br />
## Introduction

This project focuses on the problem of terrain classification for Mars rovers. This task is essential for future autonomous rover missions, as it can help rovers navigate safely and efficiently on the Martian surface. We used a dataset consisting of 35K images from Curiosity, Opportunity, and Spirit rovers with semantic segmentation labels collected through crowdsourcing. The dataset also includes 1.5K test labels annotated by the rover planners and scientists from NASA's MSL (Mars Science Laboratory) and MER (Mars Exploration Rovers) missions.


<!--- Dataset --><br />
## Dataset

The dataset we used for training and validating the terrain classification model was created specifically for this purpose. It contains approximately 326K semantic segmentation full image labels on 35K images taken by the Curiosity, Opportunity, and Spirit rovers. Each image in the dataset was labeled by 10 different people to ensure high quality and agreement of the crowdsourced labels.

![An original MSL NAVCAM image along with a high rated AI4MARS label](assets/Untitled%201.png)

<!--- Model --><br />
## Model

We utilized a deep learning-based approach for the terrain classification task. Specifically, we finetuned a U-Net architecture implemented using ResNet50. The model was trained using PyTorch Lightning, a high-level wrapper around PyTorch that simplifies the training process.


![The U-Net architecture used for semantic segmentation](assets/Untitled%202.png)


<!--- Training --><br />
## Training

The training was carried out on Kaggle, using a P100 GPU to accelerate the computations. We used a batch size of 32 and trained the model for 30 epochs on 15,000 images. We used 20% of the dataset to create a validation set, and the rest of it for the training set. Metrics were tracked using Weights and Biases.

### Loss


![Training loss per epoch](assets/Untitled%203.png)


### Accuracy


![Validation accuracy per epoch](assets/Untitled%205.png)


### Intersection over Union


![Validation IoU per epoch](assets/Untitled%206.png)


<!--- Evaluation --><br />
## Evaluation

The model was evaluated on a test set of 322 expert-annotated images with ~1.5k segmentations. These images were annotated by rover planners and scientists from NASA.

### 1 agreement


![Untitled](assets/Untitled%209.png)


### 2 agreements


![Untitled](assets/Untitled%2010.png)


### 3 agreements


![Untitled](assets/Untitled%2011.png)


<!--- Caveats and limitations --><br />
## Caveats and limitations

1. The model was trained on a subset of the available data (15,000 images). Training on a larger dataset could potentially improve the model's performance.
2. The dataset consists of images from three different rovers, each with its own camera system and specifications. This may introduce some variability in the data, which the model might need to account for during training.
3. The semantic segmentation labels in the dataset were obtained through crowdsourcing, which could introduce some noise and inconsistencies in the labels. However, each image was labeled by 10 different people to minimize this issue.

# LeNet Trafic Signal Classifier

Self Driving Car Nano degree Trafic Signal Classifier project using python, Tensorflow 1.3, openv, LetNet5 architecture

## Requirement 1: Dataset Exploration:
Dataset Summary & Visualization: Analyze and summarize the dataset used i.e, detail of Train, Validate and Test data sets. To visualize the dataset.

## Requirement 2: Design and Test a Model Architecture:
  
  2.1. Preprocessing: Describes the preprocessing techniques used and why these techniques were chosen.
  
  2.2. Model Architecture: Provide details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.
  
  2.3. Model Training: Describe how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.
  
  2.4. Solution Approach: Describe the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

## Requirement 3: Test a Model on New Images:

Get 5 new images of German Traffic signs from Web and use the model to predict them. Calculate the accuracy of the model w.r.t these 5 images. The top five softmax probabilities of the predictions on the captured images to be outputted.

# Dataset Exploration:

Image Size = 32 x 32 x 3   
Number of Training Images = 34799  
Number of Validating Images = 4410  
Number of Testing Images = 12630  

Number of classes in the dataset is 43 which can be inferred by looking for unique values in Image classified/Label values in y_train or y_valid or y_test.

Images for each class is plotted in Step 1 and Number of images in each class is plotted for y_train, y_valid and y_test.

![](/readme_images/1.jpg)

# Design and Test a Model Architecture:

**Pre Processing:** As each pixel maximum value is 255, (pixel - 128)/ 128 is the only normalization used to pre-process the color image from the dataset of each image size 32 x 32 x 3. The input images are shuffled only once. Basic LeNet architecture is implemented with small changes by adding Dropout and required accuracy is achieved.

Weights used for training are normalized by considering random values from a truncated normal distribution with mean = 0 and sigma = 0.1.

One Hot encoding for labels in y_train, y_valid and y_test is required for softmax is used. Other Hyperparameter considered are

EPOCHS = 25  
BATCH_SIZE = 100  
Learning Rate = 0.001

**Model Architecture:**

LeNet-5 architecture is implemented by adding Dropout at certain points as given below and a training accuracy of 0.953 is achieved. Dropout is a regularization technique that helps model avoid overfitting.

`CONV2D 32x32x3 -> 28x28x6`  
`relu`  
`Dropout1(0.75)`  
`max_pool -> 14x14x6`  
`CONV2D -> 10x10x16`  
`relu`  
`dropout2(0.75)`  
`max_pool -> 5x5x16`  
`flatten -> 400`  
`FullyConnected -> 120`  
`relu`  
`dropout3(0.75)`  
`FullyConnected -> 84`  
`relu`  
`dropout4(0.75)`  
`FullyConnected -> 43`

Optimizer is used for training the model by minimizing the loss. In our training pipeline Adam optimizer is used.

**Training Model:**

By training the input dataset over this final architecture project required accuracy can be achieved. Graph below shows for considered Hyperparameter and architecture how for each EPOCH the accuracy keeps increasing.

![](/readme_images/2.jpg)

**Solution Approach:**

With the initial base architecture without proper regularization technique we couldn’t achieve the required accuracy even after 1000 EPOCHs. On each step certain parameter is adjusted to obtain the final architecture as above and with just 25 epoch a high accuracy can be achieved. Different parameter and accuracy achieved is shown in following graphs.

  EPOCH 1000  
  BATCH 256  
  Learning RATE 0.001  
  Without Dropout  
  This is my first trained model Training accuracy Graph!!

![](/readme_images/3.jpg)

  EPOCH 1000  
  BATCH 256  
  RATE 0.001  
  DROPOUT1 0.5 [Only one Dropout used]  

![](/readme_images/4.jpg)

  EPOCH 1000  
  BATCH 256  
  RATE 0.001  
  2 – DROPOUT 1,2 0.5  
  Max accuracy: 0.932  

![](/readme_images/5.jpg)

  EPOCH 1000  
  BATCH 256  
  RATE_0.001  
  2 – DROPOUT 1,2 0.5  
  Max accuracy: 0.897  

![](/readme_images/6.jpg)

  EPOCH 1000  
  BATCH 128  
  RATE 0.001  
  2 - DROPOUT1,2 0.75  
  Max Accuracy : 0.937

![](/readme_images/7.jpg)

  EPOCH_1000  
  BATCH_128  
  RATE_0.001  
  2 - DROPOUT1,2 0.5  
  Max Accuracy : 0.899  

![](/readme_images/8.jpg)

  Untill now Shuffled input were not used. For the first time input shuffling is tried.  
  EPOCH_500  
  BATCH_100  
  RATE_0.001  
  DROPOUT1 0.75  
  Max Accuracy 0.93
  
![](/readme_images/9.jpg)  
  
  Shuffled input  
  EPOCH_500  
  BATCH_100  
  RATE_0.001  
  2- DROPOUT-1,2 0.75  
  Max Accuracy : 0.948

![](/readme_images/10.jpg)

In last two graphs we could see accuracy is very high right at 1st EPOCH and could reach high accuracy but still testing accuracy could not go above 0.93. After reading through few architectures then came out with used architecture in section Model Architecture with which high accuracy could be achieved in less EPOCH and testing accuracy aswell passed the requirement.

# Testing a Model on new Image from Web

Following five images are chosen for testing the model and it passed with 100% accuracy.

![](/readme_images/11.jpg)

  11 – Right-of-way at the next intersection  
  10 – No passing for vehicles over 3.5 metric tons  
  29 – Bicycles crossing  
  41 – End of no passing  
  02 – Speed limit (50km/h)

Top 5 softmax based on these images are also determined and are as below.

![](/readme_images/12.jpg)

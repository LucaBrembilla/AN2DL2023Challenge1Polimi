# Challenge 1 - dip_learners

## Project Overview
This project demonstrates the application of neural networks for recognizing plant diseases using image classification techniques. The classification task involves distinguishing between healthy and diseased plants through binary class classification. We utilized transfer learning to leverage pre-trained models and enhance the accuracy of our predictions.

## Team Members
- Me
- Matteo Miano
- Linda Frickleton
- Marianna Dragonetti

## Introduction
The project aims to address the problem of plant disease recognition using transfer learning techniques. We experimented with different neural network architectures such as VGG16, ResNet50, and ConvNextBase. The ConvNextBase architecture delivered the best performance, achieving on the test set, which was never sent to us:
- Accuracy: 87.10%
- Precision: 82.77%
- Recall: 83.42%
- F1-Score: 83.09%

## Dataset
We were provided with a dataset of 5200 images (RGB, 96x96px) of healthy and diseased plants:
- Healthy plants: 3199 (62%)
- Diseased plants: 2001 (38%)

After removing trivial outliers, the dataset was reduced to 5004 images. To balance the dataset, class weights were computed using scikit-learn.

## Model Architecture
Our model is based on the ConvNextBase architecture, pre-trained on the ImageNet dataset. The first 277 layers of the network were frozen, and fine-tuning was applied to the remaining layers to adapt the model to the plant dataset. A stack of dense layers was added to the top, along with dropout and batch normalization to prevent overfitting.

### Dense Layers
- Leaky ReLU activation for faster convergence
- Dropout and batch normalization to improve generalization and stability
- Ridge regularization to control model complexity

## Hyperparameter Tuning
We applied the following strategies to optimize the model:
- Early stopping and learning rate scheduling to avoid overfitting
- Seed setting for reproducibility
- Data augmentation techniques such as zoom, rotation, contrast, and vertical flip to improve generalization

## Results
The final model was evaluated on the test set, yielding the following results:
- **Accuracy**: 0.8710
- **Precision**: 0.8277
- **Recall**: 0.8342
- **F1-Score**: 0.8309

![Model Performance](./images/plots.png)

## Conclusion
By employing transfer learning and optimizing hyperparameters, our ConvNextBase model successfully classified plant images with high accuracy. Data augmentation and model fine-tuning proved to be key factors in enhancing the model's performance.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/LucaBrembilla/AN2DL2023Challenge1Polimi
   ```
   
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the test script:
   ```bash
   python test.py
   ```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

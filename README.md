# Rock-Paper-Scissors Image Classification 🪨📄✂️

My first Machine Learning project focusing on Computer Vision! This project classifies hand gesture images into three categories: Rock, Paper, and Scissors using a Convolutional Neural Network (CNN).

## 📋 Project Overview

This is an image classification project that uses deep learning to recognize hand gestures in the classic Rock-Paper-Scissors game. The model is trained to distinguish between three classes:
- 🪨 **Rock** - Closed fist
- 📄 **Paper** - Open hand
- ✂️ **Scissors** - Peace sign/scissors gesture

## 🎯 Learning Objectives

As my first ML project, this helped me learn:
- Computer Vision fundamentals
- Convolutional Neural Networks (CNNs)
- Image preprocessing and data augmentation
- Model training and validation
- TensorFlow/Keras framework
- Dataset splitting and management

## 🛠️ Technologies Used

- **Python** - Primary programming language
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Image processing
- **Matplotlib** - Data visualization
- **NumPy** - Numerical computations
- **split-folders** - Dataset splitting utility

## 📁 Project Structure

```
1_Image Classification_(rock_paper_scissor)/
├── README.md                                    # Project documentation
├── Submission_Dicoding_RivalMoh_Wahyudi.ipynb  # Main Jupyter notebook
└── submission_dicoding_rivalmoh_wahyudi.py      # Python script version
```

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed, then install the required packages:

```bash
pip install tensorflow
pip install split-folders
pip install matplotlib
pip install numpy
```

### Dataset

The project uses the Rock-Paper-Scissors dataset from Dicoding Academy:
- **Source**: [Dicoding Assets Repository](https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip)
- **Total Images**: ~2,188 images
- **Classes**: 3 (Rock, Paper, Scissors)
- **Split Ratio**: 60% Training, 40% Validation

## 🏗️ Model Architecture

The CNN model consists of:

```
Input Layer (100x100x3)
    ↓
Conv2D (16 filters) + ReLU + MaxPooling + Dropout(0.25)
    ↓
Conv2D (32 filters) + ReLU + MaxPooling + Dropout(0.25)
    ↓
Conv2D (64 filters) + ReLU + MaxPooling + Dropout(0.25)
    ↓
Conv2D (128 filters) + ReLU + MaxPooling + Dropout(0.25)
    ↓
Flatten
    ↓
Dense (512 neurons) + ReLU
    ↓
Dense (3 neurons) + Softmax
```

### Key Features:
- **Input Size**: 100×100 RGB images
- **Activation Function**: ReLU for hidden layers, Softmax for output
- **Regularization**: Dropout layers (0.25) to prevent overfitting
- **Optimizer**: RMSprop
- **Loss Function**: Categorical Crossentropy

## 📊 Data Preprocessing

1. **Data Augmentation** (Training set):
   - Rescaling (1./255)
   - Shear transformation (0.2)
   - Zoom (0.2)
   - Rotation (20°)
   - Horizontal flip

2. **Validation set**:
   - Only rescaling (1./255)

## 🎯 Training Configuration

- **Epochs**: 20
- **Batch Size**: 32
- **Steps per Epoch**: 32
- **Validation Steps**: 5
- **Image Size**: 100×100 pixels

## 📈 How to Run

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook Submission_Dicoding_RivalMoh_Wahyudi.ipynb
   ```

2. **Run all cells sequentially** to:
   - Download and prepare the dataset
   - Visualize sample images
   - Build and train the model
   - Test predictions on new images

3. **Alternative**: Run the Python script directly:
   ```bash
   python submission_dicoding_rivalmoh_wahyudi.py
   ```

## 🔍 Model Features

- **Real-time Prediction**: Upload any hand gesture image for classification
- **Visualization**: View sample images from each class
- **Data Splitting**: Automatic train/validation split
- **Preprocessing Pipeline**: Automated image preprocessing and augmentation

## 📝 Key Learning Points

Through this project, I learned:

1. **Computer Vision Basics**:
   - Image preprocessing techniques
   - Data augmentation importance
   - Convolutional layer functionality

2. **Deep Learning Concepts**:
   - CNN architecture design
   - Overfitting prevention with dropout
   - Loss functions and optimizers

3. **Practical Skills**:
   - Dataset management and splitting
   - Model evaluation techniques
   - TensorFlow/Keras implementation

## 🎉 Results

The model successfully classifies Rock-Paper-Scissors gestures with reasonable accuracy for a first ML project. Key achievements:

- ✅ Successfully implemented a CNN from scratch
- ✅ Applied data augmentation techniques
- ✅ Implemented proper train/validation splitting
- ✅ Created a working image classifier
- ✅ Learned fundamental computer vision concepts

## 🔄 Future Improvements

Potential enhancements for this project:
- [ ] Implement transfer learning with pre-trained models
- [ ] Add more diverse hand positions and backgrounds
- [ ] Create a web interface for real-time classification
- [ ] Experiment with different architectures (ResNet, VGG, etc.)
- [ ] Add model evaluation metrics (precision, recall, F1-score)
- [ ] Implement cross-validation

## 🙏 Acknowledgments

- **Dicoding Academy** for providing the dataset and learning platform
- **TensorFlow/Keras** community for excellent documentation
- This project was created as part of my machine learning journey

## 📧 Contact

**RivalMoh Wahyudi**
- Project: First ML/Computer Vision Project
- Focus: Image Classification with CNNs

---

*This README documents my first step into the exciting world of Machine Learning and Computer Vision! 🚀*
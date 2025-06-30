# 1DCNN Mental Workload Classifier

A machine learning project that classifies cognitive workload levels from EEG signals using Convolutional Neural Networks (CNN).

## 📋 Overview

This project analyzes EEG brain signals to determine whether a person is experiencing high or low cognitive workload. Using the STEW Dataset, we built a CNN model that can classify workload levels with visualization of brain activity patterns.

## 🎯 Project Goals

- **Binary Classification**: Distinguish between low/medium workload (ratings 4-6) and high workload (ratings 7-9)
- **EEG Visualization**: Generate signal plots and heatmaps to understand brain activity patterns
- **Model Performance**: Achieve reliable classification accuracy using deep learning

## 📊 Dataset

**STEW Dataset** (Sustained Attention to Response Task EEG Workload Dataset)
- 14 EEG channels (Emotiv headset layout)
- 64 time points per sample
- Cognitive workload ratings from 4-9
- Binary classification target (0: Low/Medium, 1: High)

## 🧠 Model Architecture

**CNN Model Features:**
- 3 Convolutional layers with MaxPooling
- Batch Normalization and Dropout for regularization
- Dense layers for final classification
- Sigmoid activation for binary output

## 📁 Repository Structure

```
├── Mental_workload_model.ipynb    # Main Jupyter notebook with complete code
├── Project_Report.pdf              # Detailed project report
├── PRESENTATION_1DCNN-Mental-Mental-Workload-Classifier.pdf              # Project presentation slides
└── README.md                       # This file
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/1DCNN-Mental-Workload-Classifier.git
   cd 1DCNN-Mental-Workload-Classifier
   ```

2. **Open the notebook**
   - Upload to Kaggle/Google Colab or run locally
   - Download the STEW Dataset files: `dataset.mat`, `rating.mat`, `class_012.mat`

3. **Run the notebook**
   - Follow the cells step by step
   - The notebook includes data preprocessing, model training, and visualization

## 📈 Key Features

- **Data Preprocessing**: Z-score normalization and proper channel alignment
- **CNN Training**: Optimized for EEG temporal patterns
- **Visualizations**: 
  - EEG signal curves for individual channels
  - Heatmaps showing signal intensity across time and channels
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC

## 🛠️ Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework
- **NumPy & Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **scikit-learn**: Model evaluation
- **SciPy**: Signal processing

## 📊 Results

The model successfully classifies cognitive workload levels with detailed performance metrics included in the notebook. Visualizations help understand which brain regions are most active during high cognitive load.

## 📚 Documentation

- **Complete Code**: `eeg-cognitive-workload.ipynb`
- **Detailed Analysis**: `project-report.pdf`
- **Presentation**: `presentation.pptx`

## 🔮 Future Improvements

- Multi-class classification for more granular workload levels
- Feature importance analysis using explainability tools
- Real-time classification capabilities
- Expanded dataset for better generalization

## 📄 License

MIT License - Feel free to use and modify this project.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any suggestions or improvements.

---

**Note**: Make sure to download the STEW Dataset files before running the notebook. The dataset is publicly available for research purposes.

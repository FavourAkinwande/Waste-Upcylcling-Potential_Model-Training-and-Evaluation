# Waste-Upcylcling-Potential Model
**Problem Statement:** 

Waste mismanagement continues to be a major issue, particularly in developing regions where large volumes remain unsorted and underutilized.
This leads to significant environmental impacts, including pollution and resource loss.
While AI tools for waste classification exist, they often lack accessibility in low-resource contexts.
This project uses an open-source waste image dataset to train a model for practical, low-cost upcycling potential prediction.

---

## Dataset & Video Presentation

- **Dataset Link**: ([ /kaggle/input/garbage-classification-v2](url))  
- **Video Presentation**: [Watch the Project Walkthrough](<[(https://youtu.be/BP1OQFgkiOg)>]

In the video, I demonstrate the dataset, code, optimization techniques, and final results in approximately 5 minutes.

---

## Dataset Overview
- **Name**: Garbage Classification V2 Dataset
- **Description**:This image dataset contains thousands of labeled waste images categorized into various types such as cardboard, glass, metal, paper, plastic, and more. It supports computer vision tasks aimed at improving waste segregation and promoting circular economy initiatives like upcycling.
- **Features**: 
  - **Categorical**: Waste Types (Cardboard, Glass, Metal, Paper, Plastic, Trash. 
  - **Numerical**:  Extracted from the images – Height, Width, Channels)
- **Target**: Upcycling Potential: High, Medium, Low (multi-class) 

---

## Project Structure

```
Project_Name/
├── notebook.ipynb                          # Kaggle notebook with all code and analysis
├── saved_models/                           # Directory containing saved model files
│   ├── Instance 1 (Default)_model.h5
│   ├── Instance 2 - Adam + Light Drop + Light L2_model.h5
│   ├── Instance 3 - RMSprop + Deeper + Low Drop_model.h5
│   ├── Instance 5 - Adam + Moderate Drop + Deep_model.h5
│   └── xgb_model.pkl
└── README.md                               # Project overview, setup instructions, and results summary
```

## Implementation Summary
I implemented:
1. **Classical ML Models**: SVM and XGBoost (with hyperparameter tuning).  
2. **Simple Neural Network**: Baseline network without optimization.  
3. **Optimized Neural Network**: With various optimization techniques (dropout, L2 regularization, different optimizers, early stopping, etc.).  
4. **Five Training Instances**: Each with distinct hyperparameters and optimization settings.

---

## Training Instances Table

| Training Instance                              | Optimizer | Learning Rate | Dropout Rate | Num Layers | L2 Reg   | Early Stopping | Epochs | Test Loss | Test Accuracy | Precision | Recall | F1 Score |
|------------------------------------------------|-----------|----------------|--------------|------------|----------|----------------|--------|-----------|----------------|-----------|--------|----------|
| Instance 1 (Default)                           | Adam      | 0.001           | 0.00         | 2          | 0.00000  | No             | 20     | 1.3966    | 0.8929         | 0.3766    | 0.3763 | 0.3763   |
| Instance 2 - Adam + Light Drop + Light L2      | Adam      | 0.0008          | 0.10         | 3          | 0.00005  | Yes            | 60     | 0.8221    | 0.7679         | 0.3712    | 0.3641 | 0.3633   |
| Instance 3 - RMSprop + Deeper + Low Drop       | RMSprop   | 0.0006          | 0.15         | 4          | 0.00000  | Yes            | 70     | 0.3748    | 0.8906         | 0.3625    | 0.3617 | 0.3619   |
| Instance 4 - SGD + Lower Drop + Light L2       | SGD       | 0.0050          | 0.10         | 3          | 0.00010  | Yes            | 80     | 1.3000    | 0.3170         | 0.1295    | 0.3333 | 0.1865   |
| Instance 5 - Adam + Moderate Drop + Deep       | Adam      | 0.0005          | 0.20         | 4          | 0.00005  | No             | 100    | 0.9167    | 0.9063         | 0.4180    | 0.4179 | 0.4173   |

---

## Discussion of Findings

1. **Which Combination Worked Better?**  
   - According to the findings, out of all the CNN configurations, Instance 5 (Adam optimizer with a learning rate of 0.0005, dropout of 0.2, and L2 regularization of 0.00005) had the highest test accuracy (90.63%) and F1-score (0.4173). It  suggests that the model's capacity to generalize and function well on unseen data worked effectively when combining light L2 regularization, deeper architecture (4 layers), and a dropout.

2. **Which Implementation Worked Better Between the ML Algorithm and Neural Network?**  
   - The  optimized CNNs outperformed the traditional machine learning models, XGBoost and SVM, in test accuracy and F1-score. Instance 5 achieved the highest performance which shows the advantage of using deeper architectures with dropout and L2 regularization.
   - XGBoost also performed well, reaching an accuracy of 65.8% and an F1-score of 0.72 for the Low Potential class. However, its overall macro average F1-score was below that of the best CNN instance. SVM had an accuracy of 64.4% but struggled more with consistency across different classes.
   - These results suggest that for image-based classification tasks, such as predicting waste upcycling potential, CNNs offer a more reliable and accurate solution, especially when optimized with regularization and tuning the hyperparemeters.

3. **Why Performance Was Improved by Optimization Techniques**
  - The application of regularization and tuning is responsible for the optimized CNNs' superior performance.
    
    By making the network learn redundant representations, dropout assisted in preventing overfitting.
    
    A more stable model was produced as a result of L2 regularization discouraging excessively intricate weight configurations.
    
    Better convergence in deeper architectures was possible by using a slower learning rate, which allowed for more controlled, gradual updates to model weights.
    
  - The consistent performance of Instance 5 on the test set demonstrated how these strategies improved the model's capacity to recognize significant patterns without learning noise.

---

## How to Run This Notebook and Load the Best Saved Model

1. **Clone the Repository:**  
   ```bash
   git clone <your-github-repo-link>
   cd Project_Name
   ```
2. **Open the Notebook:**  
   - Open `notebook.ipynb` in kaggle Notebook.  
3. **Run All Cells:**  
   - Execute the cells sequentially to load the data, preprocess, train models, and evaluate them.  
4. **Load the Best Saved Model:**  
   - For classical models (e.g., XGBoost):  
     ```python
     import joblib
     best_model = joblib.load('saved_models/xgb_model.pkl')
     ```
   - For the optimized neural network:  
     ```python
     from tensorflow.keras.models import load_model
     best_nn_model = load_model('saved_models/Instance 5 - Adam + Moderate Drop + Deep_model.h5)


     

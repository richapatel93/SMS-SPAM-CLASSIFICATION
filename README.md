### SMS Spam Classification using Machine Learning

#**Overview**

This project focuses on building a binary classification model to classify SMS as Spam or Not Spam (Ham). The goal is to demonstrate the end-to-end process of building a machine learning model, from data preprocessing and feature extraction to model training, evaluation, and interpretation of results. The project uses the Spam SMS Dataset and leverages Python libraries like Scikit-learn, Pandas, and Matplotlib.

## Key Objectives

1. **Data Preprocessing**  
   Clean and preprocess text data for machine learning.

2. **Feature Extraction**  
   Convert text data into numerical features using `CountVectorizer`.

3. **Model Training**  
   Train a **Multinomial Naive Bayes** classifier for spam detection.

4. **Model Evaluation**  
   Evaluate the model using metrics like:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC Score

5. **Visualization**  
   Plot the **ROC Curve** to visualize model performance.

6. **Interpretation**  
   Analyze the impact of **False Positives** and **False Negatives** on decision-making.

   ## Dataset

The dataset used in this project is the **Spam SMS Dataset**, which contains labeled SMS messages classified as either "spam" or "ham" (not spam). It is available on [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
## Steps Performed

1. **Data Loading and Exploration**  
   Loaded the dataset and explored its structure.

2. **Data Preprocessing**  
   Converted labels into binary values (0 for "ham" and 1 for "spam").

3. **Train-Test Split**  
   Split the dataset into training and testing sets.

4. **Feature Extraction**  
   Used `CountVectorizer` to convert text data into numerical features.

5. **Model Training**  
   Trained a **Multinomial Naive Bayes** classifier.

6. **Model Evaluation**  
   Evaluated the model using metrics like:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC-AUC Score

7. **Visualization**  
   Plotted the **ROC curve** to analyze model performance.

8. **Interpretation**  
   Analyzed the confusion matrix to understand the impact of **False Positives** and **False Negatives**.
## Key Challenges

1. **Imbalanced Dataset**  
   The dataset had more "ham" messages than "spam," which could lead to biased model performance. This was addressed by using appropriate evaluation metrics like **Precision** and **Recall**.

2. **Text Preprocessing**  
   Handling raw text data required careful preprocessing, such as:
   - Removing special characters
   - Converting text to lowercase

3. **Model Selection**  
   Choosing the right algorithm (**Naive Bayes**) for text classification and tuning hyperparameters for optimal performance.
## Results

- **Accuracy**  
  Achieved an accuracy of over **98%** on the test set.

- **Precision**  
  High precision indicates that the model correctly identified most **spam** messages.

- **Recall**  
  High recall indicates that the model minimized the number of **spam** messages incorrectly classified as **ham**.

- **ROC-AUC Score**  
  A score close to **1** indicates excellent model performance in distinguishing between **spam** and **ham**.


## Future Enhancements

1. **Advanced Text Preprocessing**  
   Incorporate techniques like **stemming**, **lemmatization**, and **stopword removal** to improve text preprocessing.

2. **Model Improvement**  
   Experiment with other algorithms like:
   - **Logistic Regression**
   - **Random Forest**
   - Deep learning models (e.g., **LSTM**)

3. **Deployment**  
   Deploy the model as a web application using **Flask** or **Streamlit** for real-time spam detection.

4. **Handling Imbalanced Data**  
   Use techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) or **class weighting** to address dataset imbalance.

5. **Hyperparameter Tuning**  
   Optimize model performance using techniques like **GridSearchCV** or **RandomizedSearchCV**.
## Learning Outcomes

Through this project, I gained hands-on experience in:

- **Data Preprocessing**  
  Cleaning and preparing text data for machine learning.

- **Feature Extraction**  
  Converting text into numerical features using **CountVectorizer**.

- **Model Training**  
  Building and training a machine learning model using **Scikit-learn**.

- **Model Evaluation**  
  Interpreting metrics like:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC

- **Visualization**  
  Plotting **ROC curves** to analyze model performance.

- **Problem-Solving**  
  Addressing challenges like **imbalanced datasets** and **text preprocessing**.

## How to Run the Code

1. **Clone the repository**  
   Run the following command to clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-classification.git


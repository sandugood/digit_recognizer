# Digit Recognizer
Handwritten digit recognizer project using Classic ML and Python.
Creation of this project was inspired by the popular among ML community 'MNIST' dataset, which has more than 50,000 unique 784-pixel pictures of handwritten digits
The process of creating this project was split into two parts.
# Exploring the dataset in the Jupyter Notebook
The main model was set to be an XGBoostClassifier (which is an upgrade to vanilla GBDT). The dataset was already preprocessed, had no missing values and had no class imbalance. 

No data preparation was needed for this particular dataset, so I just used hyperopt and GridSearchCV to perform hyperparameter tuning with cross-validation. Tuned: n_estimators (number of weak learners in the ensemble), max_depth (maximum depth of a single weak learner), gamma (minimum loss reduction needed to split the node), reg_alpha (strength of the l1 regularization), learning_rate (boosting learning rate a.k.a coefficient when adding another weak learner in the ensemble)

# Creating a GUI using Python's Tkinter module
In this step I created a user interface for drawing numbers and predicting them. For this step I used: tkinter, PIL, pyautogui, together with cv2 module. After each evalution it gives the predicted number (based on the maximum value of predict_proba method of the classifier) and the confidence level (the greatest probability of the classifier that it assigns)




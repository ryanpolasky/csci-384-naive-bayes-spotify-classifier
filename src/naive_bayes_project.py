"""
AI Programming Project – Naive Bayes Classifier
University of North Dakota – CSCI 384 AI Course | Spring 2025

Title: Predicting Hit Songs Using Naive Bayes
Total Points: 100 (+10 bonus points)

This is the main assignment script. You must complete each step where "YOUR CODE HERE" is indicated.
Use the provided helper modules (dataset_utils.py and naive_bayes_model.py) to assist you.
The NaiveBayesContinuous model is based on Artificial Intelligence: A Modern Approach, 4th US edition.
GitHub repository: https://github.com/aimacode/aima-python
"""

from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------
# STEP 1 [10 pts]: Load the Dataset
# ---------------------------------------------------------------
# - Load the CSV file 'spotify_hits.csv' (located in the 'data' folder) into a pandas DataFrame.
# - Hint: Use the load_dataset() function from dataset_utils.py. Note that you need to import the function first.

# import necessary modules and load the dataset. Hint: src/dataset_utils.py
# YOUR CODE HERE:

data =  # YOUR CODE HERE

# - Display shape of the DataFrame.
# - Hint: Use the shape attribute to get the dimensions.

print(f"Data shape: ...") # YOUR CODE HERE

# - Displace the first few rows of the DataFrame to understand its structure.
# - Hint: Use the head() method to display the first few rows.

# YOUR CODE HERE:

# ---------------------------------------------------------------
# STEP 2 [10 pts]: Create a Binary Target Column
# ---------------------------------------------------------------
# - Create a new column 'hit' from 'popularity'. A song is a hit if popularity ≥ 70; otherwise, it is not a hit.
# - Hint: Use a lambda function to create the new column. The new column should be binary (0 for not a hit, 1 for a hit).

data['hit'] = # YOUR CODE HERE

# - Delete the original 'popularity' column as it is no longer needed.
# - Hint: Use the drop() method to remove the column.

# YOUR CODE HERE:
data = # YOUR CODE HERE

# - Display unique values of the 'hit' column to verify the transformation.

# YOUR CODE HERE:


# ---------------------------------------------------------------
# STEP 3 [10 pts]: Preprocess the Dataset
# ---------------------------------------------------------------
# - Prepare the dataset for training by:
#   1. Keeping only numeric columns.
#   2. Removing rows with missing values.
# - Hint: Use select_dtypes() to select numeric columns and dropna() to remove missing values.
# - Ensure that the 'hit' column is included in the final DataFrame for target variable.

# YOUR CODE HERE:
# Select numeric columns and remove missing values.
data = # YOUR CODE HERE
data = # YOUR CODE HERE

# - Display shape of the DataFrame.
# - Hint: Use the shape attribute to get the dimensions.

print(f"data shape: ...") # YOUR CODE HERE

# ---------------------------------------------------------------
# STEP 4 [10 pts]: Train/Test Split
# ---------------------------------------------------------------
# - Split the dataset into training (80%) and testing (20%) sets.
# - Hint: Use the split_dataset() function from dataset_utils.py.

# YOUR CODE HERE:
# Split the dataset.
train_df, test_df = # YOUR CODE HERE

# - Display the shape of the training and testing DataFrames.
print(f"Train shape: {...}, Test shape: {...}") # YOUR CODE HERE

# ---------------------------------------------------------------
# STEP 5 [20 pts]: Train the Naive Bayes Model
# ---------------------------------------------------------------
# - Wrap the training DataFrame using the DataSet class (from dataset_utils.py) with 'hit' as the target.
# - Train the NaiveBayesContinuous model (from naive_bayes_model.py) using the training DataSet.

# YOUR CODE HERE:
# Create the DataSet object and train the model. Don't forget to import the NaiveBayesContinuous class.

train_dataset = # YOUR CODE HERE
model = # YOUR CODE HERE

# ---------------------------------------------------------------
# STEP 6 [20 pts]: Make Predictions and Evaluate
# ---------------------------------------------------------------
# - For each song in the test set, extract its features (all columns except 'hit') as a dictionary.
# - Use the trained model to predict the label.
# - Compare the prediction to the true 'hit' value and compute the overall accuracy.
# - Hint: Use Naive Bayes' formula to calculate the probability of each class given the features.
# - Accuracy = (Number of correct predictions) / (Total number of predictions)

# YOUR CODE HERE:
# Write your loop to predict and calculate accuracy.
correct = # YOUR CODE HERE:
total = # YOUR CODE HERE:
for ... in test_df.iterrows():
    features = # YOUR CODE HERE:
    true_label = # YOUR CODE HERE:
    predicted_label = # YOUR CODE HERE:
    if predicted_label == true_label:
        correct += 1
accuracy = # YOUR CODE HERE:
print(f"Model accuracy: {accuracy:.2f}")


# ---------------------------------------------------------------
# STEP 7 [20 pts]: Answer Conceptual Questions
# ---------------------------------------------------------------
# For each question, assign your answer (as a capital letter "A", "B", "C", or "D"). Add explanations for your choices.

# Q1 [10 pts]: Which features are most likely to influence whether a song is a hit? Explain your reasoning.
#   A. track_id and duration_ms
#   B. danceability, acousticness, and instrumentalness
#   C. popularity and tempo
#   D. artist name and genre

# Hint: Correlation analysis can help identify influential features. Sort descending by correlation with the target variable. Target variable has correlation of 1.0.

# YOUR CODE HERE:
correlations = # YOUR CODE HERE
print("Correlation of features with 'hit':")
print(correlations)

q1_answer = ""  # YOUR ANSWER HERE
q1_explanation = ""  # YOUR EXPLANATION HERE

# Q2 [5 pts]: What assumption does the Naive Bayes model make about the input features? Explain your reasoning.
#   A. They follow a uniform distribution.
#   B. They are normally distributed.
#   C. They are independent given the target class.
#   D. They are weighted by importance.
# Hint: Refer to the Naive Bayes assumption. Ref: https://en.wikipedia.org/wiki/Naive_Bayes_classifier

q2_answer = ""  # YOUR ANSWER HERE
q2_explanation = ""  # YOUR EXPLANATION HERE

# Q3 [5 pts]: What is a likely difference if a decision tree is used instead of Naive Bayes? Explain your reasoning.
#   A. The model will assume independence of features.
#   B. The model will assign probabilities instead of decision rules.
#   C. The model will create splits based on feature thresholds.
#   D. The model will always perform worse.
# Hint: Consider how decision trees work compared to Naive Bayes. Ref: https://en.wikipedia.org/wiki/Decision_tree_learning

q3_answer = ""  # YOUR ANSWER HERE
q3_explanation = ""  # YOUR EXPLANATION HERE

# ---------------------------------------------------------------
# BONUS SECTION: Advanced Analysis [10 bonus pts]
# ---------------------------------------------------------------
# BONUS Task 1 [6 pts]:
# - A. Compute and display a confusion matrix comparing the true labels to your model's predictions.
# - B. Interpret the confusion matrix. What does it tell you about the model's performance?
# - Hint: You may use sklearn.metrics.confusion_matrix. Ref: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

# - A. Compute and display a confusion matrix comparing the true labels to your model's predictions.

# YOUR CODE HERE:
y_true = ""  # YOUR CODE HERE
y_pred = ""  # YOUR CODE HERE
cm = ""  # YOUR CODE HERE
print("Confusion Matrix:")
print(cm)

# - B. Interpret the confusion matrix. What does it tell you about the model's performance?
bonus_task_1_interpretation = "" # YOUR INTERPRETATION HERE

# BONUS Task 2 [4 pts]:
# - Experiment with different thresholds for defining a hit (thresholds = [65, 70, 75, 80]).
# - Determine which threshold gives the best model accuracy.
# - Report your best threshold and the corresponding accuracy.

# - Hint:
# Try iterating over a list of possible thresholds (for example, 65, 70, 75, 80). For each threshold, update your target column 'hit' so that a song is marked as a hit if its popularity is greater than or equal to that threshold. Then, split the dataset, train your model, and compute its accuracy on the test set. Store each threshold's accuracy (for example, in a dictionary), and finally, select the threshold with the highest accuracy. Assign this best threshold and its accuracy to the variables best_threshold and best_accuracy.

# - Note: DO NOT write your code here. Only provide the best threshold and accuracy below.
best_threshold = 0  # YOUR BEST THRESHOLD HERE
best_accuracy = 0.0  # YOUR BEST ACCURACY HERE (update after testing)
print(f"Best threshold: {best_threshold}, Best accuracy: {best_accuracy:.2f}")

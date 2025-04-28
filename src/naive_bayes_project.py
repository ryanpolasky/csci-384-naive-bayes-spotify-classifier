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

from src.dataset_utils import load_dataset
import pandas as pd

data = load_dataset('data/spotify_hits.csv')  # YOUR CODE HERE

# - Display shape of the DataFrame.
# - Hint: Use the shape attribute to get the dimensions.

print(f"Data shape: {data.shape}")  # YOUR CODE HERE

# - Displace the first few rows of the DataFrame to understand its structure.
# - Hint: Use the head() method to display the first few rows.

# YOUR CODE HERE:
data.head()

# ---------------------------------------------------------------
# STEP 2 [10 pts]: Create a Binary Target Column
# ---------------------------------------------------------------
# - Create a new column 'hit' from 'popularity'. A song is a hit if popularity ≥ 70; otherwise, it is not a hit.
# - Hint: Use a lambda function to create the new column. The new column should be binary (0 for not a hit, 1 for a hit).

data['hit'] = data['popularity'].apply(lambda x: 1 if x >= 70 else 0)  # YOUR CODE HERE

# - Delete the original 'popularity' column as it is no longer needed.
# - Hint: Use the drop() method to remove the column.

data = data.drop('popularity', axis=1)  # YOUR CODE HERE

# - Display unique values of the 'hit' column to verify the transformation.

# YOUR CODE HERE:
print("Unique values in 'hit' column:", data['hit'].unique())
print("Count of each value:")
print(data['hit'].value_counts())

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
numeric_data = data.select_dtypes(include=['number'])  # Select numbers

if 'hit' not in numeric_data.columns:  # If for some reason 'hit' isn't included, add it back
    numeric_data['hit'] = data['hit']

data = numeric_data.dropna()  # Drop missing values

# - Display shape of the DataFrame.
# - Hint: Use the shape attribute to get the dimensions.

print(f"Data shape after preprocessing: {data.shape}")  # YOUR CODE HERE
print(f"Number of features (excluding target): {data.shape[1] - 1}")  # YOUR CODE HERE

# ---------------------------------------------------------------
# STEP 4 [10 pts]: Train/Test Split
# ---------------------------------------------------------------
# - Split the dataset into training (80%) and testing (20%) sets.
# - Hint: Use the split_dataset() function from dataset_utils.py.

# YOUR CODE HERE:
from src.dataset_utils import split_dataset

# Split the dataset.
train_df, test_df = split_dataset(df=data, target='hit', test_size=0.2)  # YOUR CODE HERE

# - Display the shape of the training and testing DataFrames.
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")  # YOUR CODE HERE

# ---------------------------------------------------------------
# STEP 5 [20 pts]: Train the Naive Bayes Model
# ---------------------------------------------------------------
# - Wrap the training DataFrame using the DataSet class (from dataset_utils.py) with 'hit' as the target.
# - Train the NaiveBayesContinuous model (from naive_bayes_model.py) using the training DataSet.

# YOUR CODE HERE:
# Create the DataSet object and train the model. Don't forget to import the NaiveBayesContinuous class.
from src.dataset_utils import DataSet
from src.naive_bayes_model import NaiveBayesContinuous

train_dataset = DataSet(train_df, target='hit')  # YOUR CODE HERE
model = NaiveBayesContinuous(train_dataset)  # YOUR CODE HERE

# ---------------------------------------------------------------
# STEP 6 [20 pts]: Make Predictions and Evaluate
# ---------------------------------------------------------------
# - For each song in the test set, extract its features (all columns except 'hit') as a dictionary.
# - Use the trained model to predict the label.
# - Compare the prediction to the true 'hit' value and compute the overall accuracy.
# - Hint: Use Naive Bayes' formula to calculate the probability of each class given the features.
# - Accuracy = (Number of correct predictions) / (Total number of predictions)

# YOUR CODE HERE:
correct = 0  # Counters
total = len(test_df)

# Write your loop to predict and calculate accuracy.
for index, row in test_df.iterrows():
    # Extract features as a dict (all columns except 'hit')
    features = row.drop('hit').to_dict()

    # Get the true label
    true_label = row['hit']

    # Predict the label
    predicted_label = model(features)

    # Check if prediction is correct
    if predicted_label == true_label:
        correct += 1

accuracy = correct / total  # YOUR CODE HERE.

print(f"Model accuracy: {accuracy:.2f}")  # YOUR CODE HERE.


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
correlations = data.corr()['hit'].sort_values(ascending=False)
print("Correlation of features with 'hit':")
print(correlations)

q1_answer = "B"  # YOUR ANSWER HERE
# YOUR EXPLANATION HERE
q1_explanation = """Based on the correlation analysis, audio features like danceability, acousticness, and 
instrumentalness are likely to have positive correlations with the 'hit' status than other identifiers like the track_id. 
These features represent actual musical characteristics that listeners respond to. Track_id is just an identifier with 
no meaningful correlation. Duration_ms might have some correlation but generally isn't as significant as the audio 
features. We removed popularity from the dataset (since it was used to create our target variable), and while tempo may 
have some influence, the combination of features in option B is more comprehensive in determining a song's commercial success.
"""

# Q2 [5 pts]: What assumption does the Naive Bayes model make about the input features? Explain your reasoning.
#   A. They follow a uniform distribution.
#   B. They are normally distributed.
#   C. They are independent given the target class.
#   D. They are weighted by importance.
# Hint: Refer to the Naive Bayes assumption. Ref: https://en.wikipedia.org/wiki/Naive_Bayes_classifier

q2_answer = "C"  # YOUR ANSWER HERE
# YOUR EXPLANATION HERE
q2_explanation = """
The assumption of Naive Bayes is that the features are conditionally independent given the target class. This 'naive' 
assumption simplifies the computation by allowing us to multiply individual feature probabilities. While our 
implementation also assumes that continuous features follow a normal distribution (as seen in the use of the gaussian 
function), the core naive assumption is about feature independence. The model doesn't weight features by importance, nor 
does it assume a uniform distribution across feature values.
"""

# Q3 [5 pts]: What is a likely difference if a decision tree is used instead of Naive Bayes? Explain your reasoning.
#   A. The model will assume independence of features.
#   B. The model will assign probabilities instead of decision rules.
#   C. The model will create splits based on feature thresholds.
#   D. The model will always perform worse.
# Hint: Consider how decision trees work compared to Naive Bayes. Ref: https://en.wikipedia.org/wiki/Decision_tree_learning

q3_answer = "C"  # YOUR ANSWER HERE
# YOUR EXPLANATION HERE
q3_explanation = """
Unlike Naive Bayes which treats features probabilistically, DTs operate by creating sequential splits in the data based 
on feature threshold values. Each node in a DT represents a test on a specific feature (i.e. is danceability > 0.7?), 
creating decision boundaries. Naive Bayes assumes feature independence (not DTs), and Naive Bayes actually assigns
probabilities rather than decision rules (opposite of option B). There's no guarantee that DTs would always perform 
worse than Naive Bayes, because performance depends on the specific dataset and problem.
"""

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

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import numpy as np
import os


# data_dict = pickle.load(open('./data.pickle', 'rb'))

# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# model = RandomForestClassifier()

# model.fit(x_train, y_train)

# y_predict = model.predict(x_test)

# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()



# Define the path to the dataset
DATA_DIR = '/home/saku/asl/sign-language-detector-python/data'

# Load data from the pickle file
with open('./data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

# Ensure data and labels are loaded as numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialize the Random Forest Classifier
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate the accuracy score
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Calculate overall metrics
precision = precision_score(y_test, y_predict, average='weighted') * 100
recall = recall_score(y_test, y_predict, average='weighted') * 100
f1 = f1_score(y_test, y_predict, average='weighted') * 100

print(f'Precision: {precision:.2f}%')
print(f'Recall: {recall:.2f}%')
print(f'F1 Score: {f1:.2f}%')

# Generate detailed classification report
report = classification_report(y_test, y_predict, target_names=[str(i) for i in np.unique(labels)])
print("\nDetailed Classification Report:\n")
print(report)

# Check if DATA_DIR exists and is accessible
if not os.path.exists(DATA_DIR):
    print(f"Error: The directory '{DATA_DIR}' does not exist.")
else:
    try:
        # Create a labels dictionary based on the class directories
        class_directories = os.listdir(DATA_DIR)
        if not class_directories:
            print(f"Error: No class directories found in '{DATA_DIR}'.")
        else:
            labels_dict = {int(k): v for k, v in enumerate(sorted(class_directories))}
            print("Labels Dictionary:", labels_dict)
    except Exception as e:
        print(f"Error creating labels dictionary: {e}")

# Save the trained model and labels dictionary to a pickle file
output_path = 'model.p'
try:
    with open(output_path, 'wb') as f:
        pickle.dump({'model': model, 'labels_dict': labels_dict}, f)
    print(f"Model and labels dictionary saved to {output_path}")
except Exception as e:
    print(f"Error saving the model and labels dictionary: {e}")
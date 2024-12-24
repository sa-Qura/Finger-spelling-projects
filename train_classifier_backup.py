import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

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

# --- Random Forest Classifier ---
rf_model = RandomForestClassifier()
rf_model.fit(x_train, y_train)
rf_y_predict = rf_model.predict(x_test)
rf_score = accuracy_score(y_test, rf_y_predict)
print(f'Random Forest Accuracy: {rf_score * 100:.2f}%')

# --- Support Vector Machine (SVM) ---
svm_model = SVC()
svm_model.fit(x_train, y_train)
svm_y_predict = svm_model.predict(x_test)
svm_score = accuracy_score(y_test, svm_y_predict)
print(f'SVM Accuracy: {svm_score * 100:.2f}%')

# --- Convolutional Neural Network (CNN) ---
# Reshape data for CNN (ensure it's in the format [samples, width, height, channels])
input_shape = (x_train.shape[1], 1)  # Adjust depending on your data
x_train_cnn = x_train.reshape(x_train.shape[0], *input_shape)
x_test_cnn = x_test.reshape(x_test.shape[0], *input_shape)

# Normalize data for better CNN performance
x_train_cnn = x_train_cnn / 255.0
x_test_cnn = x_test_cnn / 255.0

# Create the CNN model
cnn_model = Sequential([
    Flatten(input_shape=input_shape),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(labels)), activation='softmax')
])

# Compile and train the CNN model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(x_train_cnn, y_train, epochs=10, verbose=2)  # Adjust epochs as needed
cnn_loss, cnn_score = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)
print(f'CNN Accuracy: {cnn_score * 100:.2f}%')

# --- Save Models ---
output_path = 'model.p'
try:
    with open(output_path, 'wb') as f:
        pickle.dump({'rf_model': rf_model, 'svm_model': svm_model}, f)
    print(f"Random Forest and SVM models saved to {output_path}")

    # Save CNN model separately (Keras model)
    cnn_model.save('cnn_model.h5')
    print("CNN model saved to 'cnn_model.h5'")
except Exception as e:
    print(f"Error saving models: {e}")

# --- Labels Dictionary ---
if not os.path.exists(DATA_DIR):
    print(f"Error: The directory '{DATA_DIR}' does not exist.")
else:
    try:
        class_directories = os.listdir(DATA_DIR)
        if not class_directories:
            print(f"Error: No class directories found in '{DATA_DIR}'.")
        else:
            labels_dict = {int(k): v for k, v in enumerate(sorted(class_directories))}
            print("Labels Dictionary:", labels_dict)
    except Exception as e:
        print(f"Error creating labels dictionary: {e}")

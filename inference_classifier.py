""" import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    1: 'ka',
    2: 'kha',
    3: 'ko',
    4: 'kho',
    5: 'ngo',
    6: 'cha',
    7: 'chha',
    8: 'cho',
    9: 'chho',
    10: 'nho',
    11: 'da',
    12: 'tha1',
    13: 'do',
    14: 'tho1',
    15: 'na',
    17: 'tha2',
    18: 'to',
    19: 'tho2',
    20: 'no',
    21: 'ba',
    22: 'pha',
    23: 'po',
    24: 'pho',
    25: 'mo',
    26: 'yo',
    27: 'ro',
    28: 'lo',
    29: 'vo',
    30: 'sa',
    31: 'ha',
    32: 'la',
    33: 'or'
}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        # predicted_character = labels_dict[int(prediction[0])]
        predicted_index = int(prediction[0])
        if predicted_index in labels_dict:
            predicted_character = labels_dict[predicted_index]
        else:
            print(f"Invalid prediction index: {predicted_index}")
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
 """
import pickle
import os
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# labels_dict = model_dict['labels_dict']


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    1: 'ka',
    2: 'kha',
    3: 'ko',
    4: 'kho',
    5: 'ngo',
    6: 'cha',
    7: 'chha',
    8: 'cho',
    9: 'chho',
    10: 'nho',
    11: 'da',
    12: 'tha1',
    13: 'do',
    14: 'tho1',
    15: 'na',
    17: 'tha2',
    18: 'to',
    19: 'tho2',
    20: 'no',
    21: 'ba',
    22: 'pha',
    23: 'po',
    24: 'pho',
    25: 'mo',
    26: 'yo',
    27: 'ro',
    28: 'lo',
    29: 'vo',
    30: 'sa',
    31: 'ha',
    32: 'la',
    33: 'or'
}
# DATA_DIR = '/home/saku/asl/sign-language-detector-python/data'
# labels_dict = {int(k): v for k, v in enumerate(sorted(os.listdir(DATA_DIR)))}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    flipped_frame = cv2.flip(frame, 1)
    
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        # predicted_character = labels_dict[int(prediction[0])]
        # predicted_index = int(prediction[0]) 
        prediction_value = prediction[0].split('_')[0]  
        try:
            predicted_index = int(prediction_value)
        except ValueError:
            print(f"Error: Non-numeric prediction value '{prediction_value}'")
        if predicted_index in labels_dict:
            predicted_character = labels_dict[predicted_index]
        else:
            print(f"Invalid prediction index: {predicted_index}")
            continue
        print(f"Raw prediction: {prediction}")
        print(f"Predicted index: {predicted_index}")
        print(f"Mapped label: {labels_dict.get(predicted_index, 'Invalid')}")


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()

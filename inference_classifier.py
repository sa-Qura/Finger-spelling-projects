""" import pickle
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
    16: 'tor',
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
 """

import pickle
import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# load khmer font
khmer_font_path = '/home/saku/asl/font/KhmerOSMoul.ttf'
khmer_font = ImageFont.truetype(khmer_font_path, 32)


# labels_dict = model_dict['labels_dict']


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# labels_dict = {
#     1: 'ka',
#     2: 'kha',
#     3: 'ko',
#     4: 'kho',
#     5: 'ngo',
#     6: 'cha',
#     7: 'chha',
#     8: 'cho',
#     9: 'chho',
#     10: 'nho',
#     11: 'da',
#     12: 'tha1',
#     13: 'do',
#     14: 'tho1',
#     15: 'na',
#     16: 'tor',
#     17: 'tha2',
#     18: 'to',
#     19: 'tho2',
#     20: 'no',
#     21: 'ba',
#     22: 'pha',
#     23: 'po',
#     24: 'pho',
#     25: 'mo',
#     26: 'yo',
#     27: 'ro',
#     28: 'lo',
#     29: 'vo',
#     30: 'sa',
#     31: 'ha',
#     32: 'la',
#     33: 'or'
# }

labels_dict = {
    1: 'ក',
    2: 'ខ',
    3: 'គ',
    4: 'ឃ',
    5: 'ង',
    6: 'ច',
    7: 'ឆ',
    8: 'ជ',
    9: 'ឈ',
    10: 'ញ',
    11: 'ដ',
    12: 'ឋ',
    13: 'ឌ',
    14: 'ឍ',
    15: 'ណ',
    16: 'ត',
    17: 'ថ',
    18: 'ទ',
    19: 'ធ',
    20: 'ន',
    21: 'ប',
    22: 'ផ',
    23: 'ព',
    24: 'ភ',
    25: 'ម',
    26: 'យ',
    27: 'រ',
    28: 'ល',
    29: 'វ',
    30: 'ស',
    31: 'ហ',
    32: 'ឡ',
    33: 'អ'
}

# while True:

#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()

#     # flipped_frame = cv2.flip(frame, 1)

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         for hand_landmarks in results.multi_hand_landmarks:
#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y

#                 x_.append(x)
#                 y_.append(y)

#             for i in range(len(hand_landmarks.landmark)):
#                 x = hand_landmarks.landmark[i].x
#                 y = hand_landmarks.landmark[i].y
#                 data_aux.append(x - min(x_))
#                 data_aux.append(y - min(y_))

#         x1 = int(min(x_) * W) - 10
#         y1 = int(min(y_) * H) - 10

#         x2 = int(max(x_) * W) - 10
#         y2 = int(max(y_) * H) - 10

#         prediction = model.predict([np.asarray(data_aux)])

#         # predicted_character = labels_dict[int(prediction[0])]
#         # predicted_index = int(prediction[0]) 
#         prediction_value = prediction[0].split('_')[0]  
#         try:
#             predicted_index = int(prediction_value)
#         except ValueError:
#             print(f"Error: Non-numeric prediction value '{prediction_value}'")
#         if predicted_index in labels_dict:
#             predicted_character = labels_dict[predicted_index]
#         else:
#             print(f"Invalid prediction index: {predicted_index}")
#             continue

#         pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         draw = ImageDraw.Draw(pil_image)

#         # Draw rectangle and Khmer label
#         draw.rectangle([(x1, y1), (x2, y2)], outline="black", width=4)
#         draw.text((x1, y1 - 40), predicted_character, font=khmer_font, fill="black")

#         # Convert back to OpenCV format
#         frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

#         print(f"Raw prediction: {prediction}")
#         print(f"Predicted index: {predicted_index}")
#         print(f"Mapped label: {labels_dict.get(predicted_index, 'Invalid')}")


#         # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#         # cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
#         #             cv2.LINE_AA)

#     cv2.imshow('frame', frame)
    
#     # wait for key press and break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q') : 
#         print("quiting webcam....")
#         break


# cap.release()
# cv2.destroyAllWindows()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            data_aux = []
            x_ = []
            y_ = []

            # Collect x, y coordinates for the hand
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the data
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Calculate bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Make prediction for the hand
            prediction = model.predict([np.asarray(data_aux)])
            prediction_value = prediction[0].split('_')[0]
            try:
                predicted_index = int(prediction_value)
            except ValueError:
                print(f"Error: Non-numeric prediction value '{prediction_value}'")
                continue

            # Get the label for the predicted index
            predicted_character = labels_dict.get(predicted_index, 'Invalid')

            # Annotate the frame
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # Draw rectangle and label for the hand
            draw.rectangle([(x1, y1), (x2, y2)], outline="black", width=4)
            draw.text((x1, y1 - 40), f"Hand {hand_no + 1}: {predicted_character}", 
                      font=khmer_font, fill="black")

            # Convert back to OpenCV format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            print(f"Hand {hand_no + 1}: Raw prediction: {prediction}")
            print(f"Hand {hand_no + 1}: Predicted index: {predicted_index}")
            print(f"Hand {hand_no + 1}: Mapped label: {predicted_character}")

    # Display the annotated frame
    cv2.imshow('frame', frame)

    # Wait for key press and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting webcam...")
        break

cap.release()
cv2.destroyAllWindows()

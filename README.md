# Hand Gesture Recognition System

This Python application recognizes hand gestures using a trained deep learning model. The system captures video from a webcam, processes the frames, and uses a pre-trained model to predict gestures in real-time.

## Prerequisites

Before running the application, ensure you have the following dependencies installed:

- Python 3.x
- OpenCV (cv2)
- TensorFlow
- NumPy

You can install the required libraries using `pip`:

```bash
pip install opencv-python tensorflow numpy
```
- gesture_model.h5             # Pre-trained model for gesture recognition
- gesture_recognition.py        # Python script for gesture recognition

## Model

The model used in this application is a gesture recognition model, which should be saved in a file named `gesture_model.h5`. It is expected to output predictions for the following gestures:

- Thumbs Up
- Thumbs Down
- Victory
- No Gesture

Ensure that the `gesture_model.h5` file is located in the same directory as the script or specify its path in the `model_path` variable within the script.

## Code Explanation

### `load_model(model_path='gesture_model.h5')`
This function loads the pre-trained gesture recognition model from the specified path (`gesture_model.h5`).

### `preprocess_frame(frame)`
This function processes each frame captured from the webcam by:
- Resizing the frame to 28x28 pixels.
- Converting the frame to grayscale.
- Normalizing the frame (converting pixel values to a range between 0 and 1).

The processed frame is then reshaped to match the input shape required by the model.

### `predict_gesture(model, frame)`
This function uses the preprocessed frame to make a prediction. It:
- Passes the frame through the model.
- Extracts the predicted gesture class and its associated confidence score.
- Returns the gesture name and confidence, or "No Gesture" if the confidence is below a predefined threshold.

### `display_prediction(frame, gesture_name, confidence)`
This function overlays the predicted gesture name and its confidence level onto the video feed by displaying the text on the captured frame.

### `start_gesture_recognition(model)`
This function initializes the webcam feed and continuously captures frames. It:
- Makes predictions using `predict_gesture`.
- Displays the results in real-time on the video feed.
- Allows the user to press 'q' to quit the webcam feed.

### `main()`
The entry point of the application. It:
- Loads the pre-trained gesture recognition model.
- Starts the gesture recognition process by calling `start_gesture_recognition`.


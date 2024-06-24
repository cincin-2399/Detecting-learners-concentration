import pickle
import numpy as np

def load_xgboost_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def predict_focus(model, l_head_yaw, l_head_pitch, l_head_roll, l_gaze_yaw, l_gaze_pitch, l_gaze_roll):
    return_values = [l_head_yaw, l_head_pitch, l_head_roll, l_gaze_yaw, l_gaze_pitch, l_gaze_roll]
    single_data = np.array(return_values).reshape(1, -1)
    prediction_mapping = {0: 'Distracted', 1: 'Focused'}
    prediction_proba = model.predict(single_data)
    predicted_class = np.argmax(prediction_proba)
    return prediction_mapping[predicted_class]
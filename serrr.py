import librosa
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    
    X = StandardScaler().fit_transform(X.reshape(-1, 1)).reshape(-1)

    feature = []
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        feature.extend(mfccs)
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=np.abs(librosa.stft(X)), sr=sample_rate).T, axis=0)
        feature.extend(chroma)
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
        feature.extend(mel)
    return feature

def load_data(test_size=0.25, min_train_size=0.1):
    x, y = [], []
    count =0 
    for file in glob.glob('/Users/ajayr/Downloads/Audio_Song_Actors_01-24/*.wav'):
        file_name = os.path.basename(file)
        count=count+1
        emotion = emotions.get(file_name.split("-")[2], None)
        if emotion not in observed_emotions:
            continue
        try:
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
            y.append(emotion)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    print(count)
    if len(x) == 0:
        raise ValueError("No audio samples found.")
    
    n_samples = len(x)
    train_size = max(1.0 - test_size, min_train_size)
    return train_test_split(np.array(x), y, test_size=test_size, train_size=train_size, random_state=9)
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, report, confusion

emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

x_train, x_test, y_train, y_test = load_data(test_size=0.25)

model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,),
                      learning_rate='adaptive', max_iter=500)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy, report, confusion = evaluate_model(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", confusion)

import speech_recognition as sr
import csv
import difflib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# --- Load Expected Texts ---
def load_expected_texts(filename="expected_texts.csv"):
    expected_texts = []
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                expected_texts.append(row[0])
    except FileNotFoundError:
        print("❌ expected_texts.csv not found.")
    return expected_texts

# --- Save Results to CSV ---
def save_to_csv(data, filename="speech_results.csv"):
    header = ['Attempt', 'Recognized Text', 'Expected Text', 'Similarity Score', 'Prediction (Correct=1)', 'Actual (Correct=1)', 'Accuracy (%)']
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
    print(f"💾 Results saved to {filename}")

# --- Calculate Similarity Score ---
def calculate_similarity(text1, text2):
    return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

# --- Main Speech Recognition and ML Logic ---
def speech_to_text():
    recognizer = sr.Recognizer()
    expected_texts = load_expected_texts()
    
    if not expected_texts:
        return

    results = []
    X_train = []
    y_train = []

    # --- Collect Training Data First ---
    for idx, expected in enumerate(expected_texts):
        with sr.Microphone() as source:
            print(f"\n🎙️ Attempt {idx+1}: Please say: '{expected}'")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

            try:
                recognized = recognizer.recognize_google(audio)
                print(f"📝 You said: {recognized}")
                
                
                similarity = calculate_similarity(recognized, expected)
                length = len(recognized)
                actual_correct = 1 if similarity > 0.8 else 0  # You can adjust threshold here

                X_train.append([length, similarity])
                y_train.append(actual_correct)

                accuracy = round(similarity * 100, 2)
                results.append([idx+1, recognized, expected, round(similarity,2), '-', actual_correct, accuracy])

            except sr.UnknownValueError:
                print("❌ Could not understand the audio.")
                results.append([idx+1, "", expected, 0.0, '-', 0, 0.0])
            except sr.RequestError as e:
                print(f"🔌 Could not request results; {e}")

    # --- Train the Decision Tree ---
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print("\n🌳 Decision Tree Model Trained Successfully!")

    # --- Now Predict New Speech (optional) ---
    print("\n🔮 Now let's test with a new speech!")

    with sr.Microphone() as source:
        print("🎙️ Speak something for testing...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            test_recognized = recognizer.recognize_google(audio)
            print(f"📝 You said: {test_recognized}")
            
            test_length = len(test_recognized)
            # For testing, just compare to the first expected text
            test_similarity = calculate_similarity(test_recognized, expected_texts[0])

            prediction = clf.predict([[test_length, test_similarity]])
            print(f"✅ Predicted: {'Correct' if prediction[0]==1 else 'Incorrect'} based on model.")

        except sr.UnknownValueError:
            print("❌ Could not understand the audio.")
        except sr.RequestError as e:
            print(f"🔌 Could not request results; {e}")

    # --- Save All Results ---
    save_to_csv(results)

if __name__ == "__main__":
    speech_to_text()

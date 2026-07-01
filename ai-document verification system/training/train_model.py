import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/ai_document_verification_dataset.csv")

X = df["comments"]
y = df["verification_status"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = RandomForestClassifier()
model.fit(X, y_encoded)

pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(encoder, open("models/label_encoder.pkl", "wb"))

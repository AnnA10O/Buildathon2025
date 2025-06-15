import os
import pandas as pd
import numpy as np
import joblib
import re

# ========== TEXT MODEL: Symptoms to Disease ==========

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Load and preprocess text data
data = pd.read_csv('Symptom2Disease.csv')
data.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = re.findall(r'\b[a-zA-Z]+\b', str(text).lower())
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

preprocessed_symptoms = data['text'].apply(preprocess_text)

# Vectorize and train KNN
tfidf_vectorizer = TfidfVectorizer(max_features=1500)
tfidf_features = tfidf_vectorizer.fit_transform(preprocessed_symptoms)
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(tfidf_features, data['label'])

# Save text model artifacts
os.makedirs('../model', exist_ok=True)
joblib.dump(tfidf_vectorizer, '../model/tfidf_vectorizer.pkl')
joblib.dump(clf, '../model/knn_model.pkl')
print("Text model trained and saved.")

# ========== IMAGE MODEL: Skin Lesion Classification from CSV ==========

import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from sklearn.model_selection import train_test_split

# Paths and parameters
ISIC_DATA_DIR = '../isic_images'  # directory where images are stored
METADATA_CSV = r'C:\Users\Bhargav\PycharmProjects\Buildthon2024\symptom2disease\backend\isic_metadata.csv'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# === DEDUPLICATE AND SORT CSV ===
metadata = pd.read_csv(METADATA_CSV)
metadata = metadata.sort_values(by='isic_id')
metadata = metadata.drop_duplicates(subset=['isic_id'], keep='first')

# Use the correct columns and drop rows where diagnosis_1 or image_url is missing
metadata = metadata[['isic_id', 'diagnosis_1', 'image_url']].dropna(subset=['diagnosis_1', 'image_url'])
metadata.rename(columns={'diagnosis_1': 'diagnosis'}, inplace=True)

# Create full image paths and filter to existing images
metadata['filepath'] = metadata['isic_id'].apply(
    lambda x: os.path.join(ISIC_DATA_DIR, f"{x}.jpg")
)
metadata = metadata[metadata['filepath'].apply(os.path.exists)]

print(f"Number of unique images with files: {len(metadata)}")

# Split into train/validation
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    metadata,
    test_size=0.2,
    stratify=metadata['diagnosis'],
    random_state=42
)

# Data generators using flow_from_dataframe
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filepath',
    y_col='diagnosis',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)
val_gen = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filepath',
    y_col='diagnosis',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse'
)

# Save class names for later use
num_classes_img = len(train_gen.class_indices)
class_names_img = list(train_gen.class_indices.keys())
joblib.dump(class_names_img, '../model/image_class_names.pkl')

# Build MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes_img, activation='softmax')(x)
image_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model for initial training
base_model.trainable = False
image_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

# Train initial layers
image_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

# Fine-tune all layers
base_model.trainable = True
image_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
image_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Save trained image model
image_model.save('model/skin_lesion_model.h5')
print("ISIC-trained image model saved.")
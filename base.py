# =========================
# General libraries
# =========================
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

print("GPU Available:", tf.config.list_physical_devices('GPU'))

# =========================
# CNN MODEL
# =========================
model = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(224,224,3)),
    Conv2D(32, (3,3), activation="relu"),
    MaxPool2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPool2D((2,2)),

    Conv2D(128, (3,3), activation="relu"),
    MaxPool2D((2,2)),

    Dropout(0.25),

    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.25),
    Dense(1, activation="sigmoid")   # Binary output
])

# 🔴 IMPORTANT: optimize for recall
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision")
    ]
)

model.summary()

# =========================
# DATA GENERATORS
# =========================
def train_generator(path):
    gen = ImageDataGenerator(
        rescale=1/255,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )
    return gen.flow_from_directory(
        path,
        target_size=(224,224),
        batch_size=32,
        class_mode="binary"
    )

def test_generator(path):
    gen = ImageDataGenerator(rescale=1/255)
    return gen.flow_from_directory(
        path,
        target_size=(224,224),
        batch_size=32,
        class_mode="binary",
        shuffle=False   # IMPORTANT
    )

trainingData   = train_generator("data/Training")
validationData = test_generator("data/Validation")
testingData    = test_generator("data/Testing")

print("Class indices:", trainingData.class_indices)

# =========================
# CALLBACKS (medical-safe)
# =========================
es = EarlyStopping(
    monitor="val_recall",
    patience=4,
    mode="max",
    verbose=1,
    restore_best_weights=True
)

mc = ModelCheckpoint(
    filepath="./bestModel.keras",
    monitor="val_recall",
    save_best_only=True,
    mode="max",
    verbose=1
)

# =========================
# CLASS WEIGHTS (CRITICAL)
# =========================
class_weight = {
    0: 1.0,   # notumor
    1: 3.0    # tumor (more important)
}

# =========================
# TRAINING
# =========================
history = model.fit(
    trainingData,
    epochs=30,
    validation_data=validationData,
    callbacks=[es, mc],
    class_weight=class_weight
)

# =========================
# TESTING (CORRECT WAY)
# =========================
model = load_model("./bestModel.keras")

y_true = testingData.classes
y_prob = model.predict(testingData)

# 🔥 LOWER THRESHOLD FOR CANCER
THRESHOLD = 0.3
y_pred = (y_prob >= THRESHOLD).astype(int).ravel()

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_true, y_pred))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true,
    y_pred,
    target_names=testingData.class_indices.keys()
))

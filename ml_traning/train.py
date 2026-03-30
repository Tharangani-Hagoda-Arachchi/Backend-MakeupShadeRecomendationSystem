import os
import ssl
import numpy as np
import pandas as pd
from preprocess import preprocess_image
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Fix SSL certificate issue (macOS Python 3.11)
ssl._create_default_https_context = ssl._create_unverified_context

# Fixed random seeds
tf.random.set_seed(42)
np.random.seed(42)

LABELS       = ["dryness", "redness", "acne", "dark_circles", "oily_zones"]
IMAGE_FOLDER = "dataset/images"
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
EPOCHS       = 100

# Load and preprocess images
df = pd.read_csv("dataset/label.csv")

# Remove rows whose image file does not exist on disk
df = df[df["image"].apply(
    lambda fn: os.path.isfile(os.path.join(IMAGE_FOLDER, fn))
)].sample(frac=1, random_state=42).reset_index(drop=True)

X, y = [], []
for _, row in df.iterrows():
    img = preprocess_image(os.path.join(IMAGE_FOLDER, row["image"]))
    if img is not None:
        X.append(img)
        y.append([int(row[l]) for l in LABELS])

X = np.array(X, dtype="float32")
y = np.array(y, dtype="float32")


print(f"Dataset: {len(X)} images  |  shape {X.shape}")

# Train / val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y[:, 2]
)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

# Class weights (fix imbalance)
pos_counts  = y_train.sum(axis=0).clip(min=1)
neg_counts  = len(y_train) - pos_counts
pos_weights = (neg_counts / pos_counts).astype("float32")
print("Per-label positive weights:", dict(zip(LABELS, pos_weights.round(2))))

LABEL_SMOOTH = 0.05

def smooth_weighted_bce(y_true, y_pred):
    y_smooth = y_true * (1.0 - LABEL_SMOOTH) + 0.5 * LABEL_SMOOTH
    pw  = tf.constant(pos_weights, dtype=tf.float32)
    bce = tf.keras.backend.binary_crossentropy(y_smooth, y_pred)
    w   = y_true * pw + (1.0 - y_true)
    return tf.reduce_mean(w * bce)

# tf.data pipelines with strong augmentation
AUTO = tf.data.AUTOTUNE

def augment(image, label):
    # Spatial
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    k     = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    # Zoom via crop
    image = tf.image.resize_with_crop_or_pad(image, 256, 256)
    image = tf.image.random_crop(image, [224, 224, 3])
    # Color — critical for skin tone variation
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.75, 1.25)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_hue(image, 0.05)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(len(X_train), seed=42)
    .map(augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Improved custom CNN

L2 = 1e-4

def conv_block(x, filters, dropout=0.0):
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(L2)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        kernel_regularizer=regularizers.l2(L2)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    if dropout:
        x = layers.SpatialDropout2D(dropout)(x)
    return x

inputs = layers.Input(shape=(224, 224, 3))
x = conv_block(inputs, 32,  dropout=0.15)
x = conv_block(x,      64,  dropout=0.15)
x = conv_block(x,      128, dropout=0.25)
x = conv_block(x,      256, dropout=0.25)
x = layers.GlobalAveragePooling2D()(x)

# Dense head with BatchNorm
x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(L2))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(L2))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(L2))(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(5, activation="sigmoid")(x)

model = models.Model(inputs, outputs, name="skin_multilabel_cnn_v2")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        weight_decay=1e-4
    ),
    loss=smooth_weighted_bce,
    metrics=[
        tf.keras.metrics.AUC(multi_label=True, name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)
model.summary()

# Callbacks
def cosine_anneal(epoch, lr):
    """Halve LR every 25 epochs to escape local minima."""
    if epoch > 0 and epoch % 25 == 0:
        return lr * 0.5
    return lr

cbs = [
    callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max",
        factor=0.5, patience=6,
        min_lr=1e-7, verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_auc", mode="max",
        patience=20,              # increased from 12 → more patience
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_auc",
        mode="max", save_best_only=True, verbose=1
    ),
    callbacks.TensorBoard(log_dir="logs/", histogram_freq=1),
    callbacks.LearningRateScheduler(cosine_anneal, verbose=0),
]

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=cbs,
)

#  Evaluate
print("\n── Validation metrics (best weights restored) ──")
model.evaluate(val_ds)

# Threshold tuning & classification report
y_val_pred = model.predict(val_ds)

thresholds = []
for i, label in enumerate(LABELS):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.91, 0.05):
        preds = (y_val_pred[:, i] >= t).astype(int)
        f1    = f1_score(y_val[:, i], preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    thresholds.append(best_t)
    print(f"  {label:<14}  best threshold: {best_t:.2f}  F1: {best_f1:.3f}")

y_pred = (y_val_pred >= np.array(thresholds)).astype(int)

print("\n── Per-label classification report ──")
print(classification_report(y_val, y_pred, target_names=LABELS, zero_division=0))

# Per-label AUC
try:
    aucs = roc_auc_score(y_val, y_val_pred, average=None)
    print("── Per-label AUC ──")
    for label, auc_val in zip(LABELS, aucs):
        print(f"  AUC {label:<14}: {auc_val:.3f}")
    print(f"\n  Mean AUC       : {np.mean(aucs):.3f}")
except ValueError as e:
    print("AUC not computable:", e)

#  Save model and thresholds
model.save("model.keras")
np.save("best_thresholds.npy", np.array(thresholds))
print("\nSaved → model.keras")
print("Saved → best_thresholds.npy")
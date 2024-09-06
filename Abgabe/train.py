import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras
import tensorflow as tf

def make_classifier():
    input_shape = (32, 32, 3)
    classifier = keras.models.Sequential()

    classifier.add(keras.layers.InputLayer(input_shape=input_shape))

    # Data Augmentation
    classifier.add(keras.layers.RandomFlip("horizontal"))
    classifier.add(keras.layers.RandomRotation(0.1))
    classifier.add(keras.layers.RandomZoom(0.1))
    classifier.add(keras.layers.RandomBrightness(0.2))
    classifier.add(keras.layers.RandomContrast(0.2))

    # Conv-BatchNorm-ReLU-Pool Block 1
    classifier.add(keras.layers.Convolution2D(32, (3, 3), padding="same", name="Conv2D_1"))
    classifier.add(keras.layers.BatchNormalization())
    classifier.add(keras.layers.ReLU(name="activation_1"))
    classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2), name="pooling_1"))

    # Conv-BatchNorm-ReLU-Pool Block 2
    classifier.add(keras.layers.Convolution2D(64, (3, 3), padding="same", name="Conv2D_2"))
    classifier.add(keras.layers.BatchNormalization())
    classifier.add(keras.layers.ReLU(name="activation_2"))
    classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2), name="pooling_2"))

    # Conv-BatchNorm-ReLU-Pool Block 3
    classifier.add(keras.layers.Convolution2D(128, (3, 3), padding="same", name="Conv2D_3"))
    classifier.add(keras.layers.BatchNormalization())
    classifier.add(keras.layers.ReLU(name="activation_3"))
    classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2), name="pooling_3"))

    # Zusätzlicher Conv-BatchNorm-ReLU-Pool Block
    classifier.add(keras.layers.Convolution2D(256, (3, 3), padding="same", name="Conv2D_4"))
    classifier.add(keras.layers.BatchNormalization())
    classifier.add(keras.layers.ReLU(name="activation_4"))
    classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2), name="pooling_4"))

    # Flatten and Dense Layers
    classifier.add(keras.layers.Flatten(name="flatten_1"))
    classifier.add(keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
    classifier.add(keras.layers.BatchNormalization())
    classifier.add(keras.layers.Dropout(0.5))
    classifier.add(keras.layers.Dense(num_classes, activation="softmax"))

    # Cosine Decay Learning Rate Schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps=100*len(train_ds)
    )

    # Optimizer mit Gradient Clipping
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    def focal_loss(gamma=2., alpha=.25):
        def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(tf.clip_by_value(pt_1, 1e-8, 1.0)))
        return focal_loss_fixed
    
    classifier.compile(
    optimizer=optimizer,
    loss=focal_loss(),
    metrics=["accuracy"]
    )

    return classifier

def make_test_predictions(classifier, test_ds):
    # Make predictions on the test set
    y_test_pred = classifier.predict(test_ds)
    y_test_pred = y_test_pred.argmax(axis=1)

    # Sort the predictions by image id in ascending order
    y_test_pred = y_test_pred[
        sorted(
            range(len(y_test_pred)),
            key=lambda x: int(test_ds.file_paths[x].split("/")[-1].split(".")[0]),
        )
    ]

    return y_test_pred

def save_predictions(y_test_pred):
    # Save the predictions as csv, with id,class headers
    with open("predictions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "class"])
        for i, pred in enumerate(y_test_pred):
            writer.writerow([i, pred])


if __name__ == "__main__":
    # Load the training data and take a subset of 20% for validation and hyperparameter tuning
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        "./data/train",
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=32,  # Reduced batch size
        image_size=(32, 32),
        interpolation="bilinear",
        data_format="channels_last",
        validation_split=0.2,
        subset="both",
        shuffle=True,
        seed=42,
        verbose=True,
    )

    # Load the test data for making predictions
    test_ds = keras.utils.image_dataset_from_directory(
        "./data/test",
        labels=None,
        color_mode="rgb",
        batch_size=32,  # Reduced batch size
        image_size=(32, 32),
        interpolation="bilinear",
        data_format="channels_last",
        shuffle=False,
        verbose=True,
    )

    # save the class names for later, we will need them to plot the confusion matrix
    class_names = train_ds.class_names
    num_classes = len(class_names)

    classifier = make_classifier()

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=30, restore_best_weights=True
    )

    # Train the model
    history = classifier.fit(
        train_ds,
        epochs=200,  # Increased number of epochs
        validation_data=val_ds,
        callbacks=[early_stopping]
    )

    # Save the model
    classifier.save("classifier.keras")

    # Make predictions on the test set
    y_test_pred = make_test_predictions(classifier, test_ds)

    # Save the predictions
    save_predictions(y_test_pred)

    # Evaluate the model on the entire validation dataset
    y_true = []
    y_val_pred = []

    for x, y in val_ds:
        y_true.extend(y.numpy().argmax(axis=1))
        y_val_pred.extend(classifier.predict(x).argmax(axis=1))

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_val_pred = np.array(y_val_pred)

    # print the classification report
    print(classification_report(y_true, y_val_pred, target_names=class_names))

    # Save a confusion matrix plot
    conf_matrix = confusion_matrix(y_true, y_val_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=class_names
    )
    disp.plot()
    plt.savefig("confusion_matrix.png")

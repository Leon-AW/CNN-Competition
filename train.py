import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras

def make_classifier():
    input_shape = (32, 32, 3)
    classifier = keras.models.Sequential()

    classifier.add(keras.layers.InputLayer(input_shape=input_shape))

    # Data Augmentation
    classifier.add(keras.layers.RandomFlip("horizontal"))
    classifier.add(keras.layers.RandomRotation(0.1))
    classifier.add(keras.layers.RandomZoom(0.1))
    

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

    # Flatten and Dense Layers
    classifier.add(keras.layers.Flatten(name="flatten_1"))
    classifier.add(keras.layers.Dense(512, activation="relu"))
    classifier.add(keras.layers.BatchNormalization())
    classifier.add(keras.layers.Dropout(0.5))
    classifier.add(keras.layers.Dense(num_classes, activation="softmax"))

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    classifier.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
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

# Learning Rate Schedule
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 75:
        lr *= 0.1
    elif epoch > 100:
        lr *= 0.01
    return lr


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
    lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True
    )

    # Train the model
    history = classifier.fit(
        train_ds,
        epochs=100,  # Increased number of epochs
        validation_data=val_ds,
        callbacks=[lr_scheduler, early_stopping]
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

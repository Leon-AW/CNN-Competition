import csv

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tensorflow import keras


def make_classifier():
    # Define the classifier model
    input_shape = (32, 32, 3)
    classifier = keras.models.Sequential()

    classifier.add(keras.layers.InputLayer(input_shape=input_shape))

    # Conv-ReLU-Pool Block 1
    classifier.add(
        keras.layers.Convolution2D(32, (3, 3), padding="same", name="Conv2D_1")
    )
    classifier.add(keras.layers.ReLU(name="activation_1"))
    classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2), name="pooling_1"))

    # Conv-ReLU-Pool Block 2
    classifier.add(
        keras.layers.Convolution2D(64, (3, 3), padding="same", name="Conv2D_2")
    )
    classifier.add(keras.layers.ReLU(name="activation_2"))
    classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2), name="pooling_2"))

    # Conv-ReLU-Pool Block 3
    classifier.add(
        keras.layers.Convolution2D(128, (3, 3), padding="same", name="Conv2D_3")
    )
    classifier.add(keras.layers.ReLU(name="activation_3"))
    classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2), name="pooling_3"))

    # Conv-ReLU-Pool Block 4
    classifier.add(
        keras.layers.Convolution2D(256, (3, 3), padding="same", name="Conv2D_4")
    )
    classifier.add(keras.layers.ReLU(name="activation_4"))
    classifier.add(keras.layers.MaxPool2D(pool_size=(2, 2), name="pooling_4"))

    # Flatten and Dense Layer
    classifier.add(keras.layers.Flatten(name="flatten_1"))
    classifier.add(keras.layers.Dense(1024, activation="relu"))
    classifier.add(
        keras.layers.Dense(num_classes, activation="softmax"),
    )

    # Compile the model, here we set an optimizer, loss function and evaluation metric
    classifier.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return classifier


def make_test_predictions(classifier, test_ds):
    # Make predictions on the test set
    y_test_pred = classifier.predict(test_ds)
    y_test_pred = y_test_pred.argmax(axis=1)

    # the image data loader reads files as
    # './data/test/0.png',
    #  './data/test/1.png',
    #  './data/test/10.png',
    #  './data/test/100.png',
    #  './data/test/1000.png',
    #  './data/test/1001.png',
    #  './data/test/1002.png',
    # so we need to sort them to get the correct order for the predictions

    # sort the predictions by image id in ascending order,
    # 0.png, 1.png, 2.png, ...
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
        batch_size=1024,
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
        batch_size=1500,
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

    # Train the model
    history = classifier.fit(train_ds, epochs=10, validation_data=val_ds)

    # Save the model
    classifier.save("classifier.keras")

    # Make predictions on the test set
    y_test_pred = make_test_predictions(classifier, test_ds)

    # Save the predictions
    save_predictions(y_test_pred)

    # Evaluate the model
    # first, get the true labels and the predicted labels
    _, y_true = val_ds.as_numpy_iterator().next()
    y_val_pred = classifier.predict(val_ds)

    # convert the one-hot encoded labels back to integer ids
    y_true = y_true.argmax(axis=1)
    y_val_pred = y_val_pred.argmax(axis=1)

    # print the classification report
    print(classification_report(y_true, y_val_pred, target_names=class_names))

    # Save a confusion matrix plot
    conf_matrix = confusion_matrix(y_true, y_val_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=class_names
    )
    disp.plot()
    plt.savefig("confusion_matrix.png")

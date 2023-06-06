import os
import random
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot import PlotLossesKeras
# from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from sklearn.metrics import precision_score, recall_score, confusion_matrix, RocCurveDisplay, roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt


# Koncepcja: Mamy folder o nazwie piłki, a w tym folderze:
#           * 18 folderów
#           * każdy po 30 zdj
#           * każdy odpowiada jednemu typowi piłki na jednym tle
#           * nazwy jak określiliśmy w docsie, ale bez polskich znaków i z "_" pomiędzy slowami
#               + na końcu 1 słowo okreslające powierzchnię np. pilka_do_noznej_trawa
#           * podział ze względu na to, aby zbiory val i test były zbilansowane pod względem zdjęć
#               z każdej kategorii i z każdego tła


# Funkcja do tworzenia setu walidacyjnego i testowego
def build_subset(image_array, label_array, val_images, val_labels, size):
    random_indices = random.sample(range(len(image_array)), size)

    for i in random_indices:
        val_images.append(image_array[i])
        val_labels.append(label_array[i])

    image_array_removed = [image_array[index] for index in range(len(image_array)) if index not in random_indices]

    label_array = np.array(label_array)
    label_array = np.delete(label_array, random_indices)
    label_array = label_array.tolist()

    return image_array_removed, label_array, val_images, val_labels


# Funkcja aby ujednolicić etykiety na koniec, pozbywając się członu trawa, stol, podloga
def unify_labels(labels_array):
    for i in range(len(labels_array)):
        label_subs = list(labels_array[i].split('_'))
        if label_subs.__contains__('podloga') or label_subs.__contains__('trawa') or label_subs.__contains__('stol'):
            labels_array[i] = '_'.join(label_subs[:-1])


# Funkcja do normalizacji danych
def normalize_data(img):
    return tf.cast(img, tf.float32) / 255.


def flip_image(img, direction):
    return cv2.flip(img, direction)


def adjust_brightness(img, delta):
    return tf.image.adjust_brightness(img, delta=delta)


def rotate_image(img, angle_range_left, angle_range_right):
    angle = np.random.randint(angle_range_left, angle_range_right)
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def zoom_image(img, zoom_factor):
    h, w = img.shape[:2]
    new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
    center_x, center_y = int(w / 2), int(h / 2)
    cropped_img = img[center_y - new_h:center_y + new_h, center_x - new_w:center_x + new_w]
    return cv2.resize(cropped_img, (w, h))


def reshape_data_to_4dim(image, label):
    image = tf.expand_dims(image, 0)
    label = tf.expand_dims(label, 0)
    return image, label


def create_label_map(labels):
    label_map = {}
    label_count = 0

    for label in labels:
        if label not in label_map:
            label_map[label] = label_count
            label_count += 1
    return label_map


def convert_to_one_hot(numbers, num_classes):
    n = len(numbers)
    one_hot = np.zeros((n, num_classes))
    for i, number in enumerate(numbers):
        one_hot[i, number] = 1
    return one_hot


# pobranie folderów=klas z folderu piłki
def main():
    directories_classes = []

    for root, dirs, _ in os.walk("./pilki/pilki", topdown=False):
        for name in dirs:
            directories_classes.append(name)

    val_images_split1 = []
    val_labels_split1 = []
    val_images_split2 = []
    val_labels_split2 = []
    val_images_split3 = []
    val_labels_split3 = []

    test_images_split1 = []
    test_labels_split1 = []
    test_images_split2 = []
    test_labels_split2 = []
    test_images_split3 = []
    test_labels_split3 = []

    train_images_split1 = []
    train_labels_split1 = []
    train_images_split2 = []
    train_labels_split2 = []
    train_images_split3 = []
    train_labels_split3 = []

    # pętla po pobranych folderach/klasach
    for dir in directories_classes:
        image_array_split1 = []
        label_array_split1 = []
        image_array_split2 = []
        label_array_split2 = []
        image_array_split3 = []
        label_array_split3 = []
        # pętla po kolejnych obrazach w danym folderze
        for filename in os.listdir('./pilki/pilki/' + dir):
            if filename.endswith('.jpg'):
                # print(f'{dir}/{filename}')
                img = cv2.imread('./pilki/pilki/' + dir + '/' + filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))  # często taki format w modelach

                # dodaj obraz i etykietę (równą nazwie folderu) do tymczasowych tablic
                image_array_split1.append(img)
                label_array_split1.append(dir)
                image_array_split2.append(img)
                label_array_split2.append(dir)
                image_array_split3.append(img)
                label_array_split3.append(dir)

                # augemntacja danych - plus 4x tyle zdjęć
                # flip pionowy względem osi x
                flipped_img = flip_image(img, 0)
                # zmiana jasności (przyciemnienie)
                bright_img = adjust_brightness(img, -0.4)
                # obrót obrazu o losowy kąt z zakresu (-90, 90) stopni
                rotated_img = rotate_image(img, -90, 90)
                # zoom x 2
                zoomed_img = zoom_image(img, 2.2)

                # dodaj nowo postałe dane do tablic
                image_array_split2.append(flipped_img)
                label_array_split2.append(dir)
                image_array_split2.append(bright_img)
                label_array_split2.append(dir)
                image_array_split2.append(rotated_img)
                label_array_split2.append(dir)
                image_array_split2.append(zoomed_img)
                label_array_split2.append(dir)

                image_array_split3.append(flipped_img)
                label_array_split3.append(dir)
                image_array_split3.append(bright_img)
                label_array_split3.append(dir)
                image_array_split3.append(rotated_img)
                label_array_split3.append(dir)
                image_array_split3.append(zoomed_img)
                label_array_split3.append(dir)

        # wywołaj funkcję losowo przydzielającą do val i test setów
        X_train_val1, X_test1, y_train_val1, y_test1 = train_test_split(image_array_split1, label_array_split1,
                                                                        test_size=0.1)
        X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train_val1, y_train_val1, test_size=0.2)

        X_train_val2, X_test2, y_train_val2, y_test2 = train_test_split(image_array_split2, label_array_split2,
                                                                        test_size=0.1)
        X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train_val2, y_train_val2, test_size=0.2)

        X_train_val3, X_test3, y_train_val3, y_test3 = train_test_split(image_array_split3, label_array_split3,
                                                                                test_size=0.1)
        X_train3, X_val3, y_train3, y_val3 = train_test_split(image_array_split3, label_array_split3, test_size=0.2)

        # rozszerz listę train o dane z tymczasowych tablic po usunięciu elementów przydzielonych do val i test
        train_images_split1 = train_images_split1 + X_train1
        train_labels_split1 = train_labels_split1 + y_train1
        val_images_split1 = val_images_split1 + X_val1
        val_labels_split1 = val_labels_split1 + y_val1
        test_images_split1 = test_images_split1 + X_test1
        test_labels_split1 = test_labels_split1 + y_test1

        train_images_split2 = train_images_split2 + X_train2
        train_labels_split2 = train_labels_split2 + y_train2
        val_images_split2 = val_images_split2 + X_val2
        val_labels_split2 = val_labels_split2 + y_val2
        test_images_split2 = test_images_split2 + X_test2
        test_labels_split2 = test_labels_split2 + y_test2

        train_images_split3 = train_images_split3 + X_train3
        train_labels_split3 = train_labels_split3 + y_train3
        train_images_split3 = train_images_split3 + X_val3
        train_labels_split3 = train_labels_split3 + y_val3
        val_images_split3 = val_images_split3 + X_val3
        val_labels_split3 = val_labels_split3 + y_val3
        test_images_split3 = test_images_split3 + X_test3
        test_labels_split3 = test_labels_split3 + y_test3

    # zamień listy na numpy arrays
    train_images_split1 = np.array(train_images_split1)
    val_images_split1 = np.array(val_images_split1)
    test_images_split1 = np.array(test_images_split1)

    train_images_split2 = np.array(train_images_split2)
    val_images_split2 = np.array(val_images_split2)
    test_images_split2 = np.array(test_images_split2)

    train_images_split3 = np.array(train_images_split3)
    val_images_split3 = np.array(val_images_split3)
    test_images_split3 = np.array(test_images_split3)

    # pozbądź się nazw tła z etykiet
    unify_labels(train_labels_split1)
    unify_labels(val_labels_split1)
    unify_labels(test_labels_split1)
    unify_labels(train_labels_split2)
    unify_labels(val_labels_split2)
    unify_labels(test_labels_split2)
    unify_labels(train_labels_split3)
    unify_labels(val_labels_split3)
    unify_labels(test_labels_split3)

    # konwertuj etykiety na indexy liczbowe a następnie na wektory One-Hot-Encoding
    labels = list(set(train_labels_split1))
    label_map = create_label_map(labels)
    train_labels_split1_index = [label_map[label] for label in train_labels_split1]
    train_labels_split1_one_hot = convert_to_one_hot(train_labels_split1_index, len(labels))

    val_labels_split1_index = [label_map[label] for label in val_labels_split1]
    val_labels_split1_one_hot = convert_to_one_hot(val_labels_split1_index, len(labels))

    test_labels_split1_index = [label_map[label] for label in test_labels_split1]
    test_labels_split1_one_hot = convert_to_one_hot(test_labels_split1_index, len(labels))

    train_labels_split2_index = [label_map[label] for label in train_labels_split2]
    train_labels_split2_one_hot = convert_to_one_hot(train_labels_split2_index, len(labels))

    val_labels_split2_index = [label_map[label] for label in val_labels_split2]
    val_labels_split2_one_hot = convert_to_one_hot(val_labels_split2_index, len(labels))

    test_labels_split2_index = [label_map[label] for label in test_labels_split2]
    test_labels_split2_one_hot = convert_to_one_hot(test_labels_split2_index, len(labels))

    train_labels_split3_index = [label_map[label] for label in train_labels_split3]
    train_labels_split3_one_hot = convert_to_one_hot(train_labels_split3_index, len(labels))

    val_labels_split3_index = [label_map[label] for label in val_labels_split3]
    val_labels_split3_one_hot = convert_to_one_hot(val_labels_split3_index, len(labels))

    test_labels_split3_index = [label_map[label] for label in test_labels_split3]
    test_labels_split3_one_hot = convert_to_one_hot(test_labels_split3_index, len(labels))

    # znormalizuj dane
    # for i in range(len(train_images_split2)):
    #     train_images_split2[i] = normalize_data(train_images_split2[i])
    # for i in range(len(val_images_split2)):
    #     val_images_split2[i] = normalize_data(val_images_split2[i])
    # for i in range(len(test_images_split2)):
    #     test_images_split2[i] = normalize_data(test_images_split2[i])
    #
    # for i in range(len(train_images_split3)):
    #     train_images_split3[i] = normalize_data(train_images_split3[i])
    # for i in range(len(val_images_split3)):
    #         val_images_split3[i] = normalize_data(val_images_split3[i])
    # for i in range(len(test_images_split3)):
    #     test_images_split3[i] = normalize_data(test_images_split3[i])

    # Define the model
    input_shape = (224, 224, 3)
    num_classes = len(labels)
    # model = Sequential([
    #     # Convolutional layers
    #     Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(256, (3, 3), activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Conv2D(512, (3, 3), activation='relu'),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     # Flatten layer
    #     Flatten(),
    #     # Dense layers with dropout
    #     Dense(512, activation='relu'),
    #     Dense(256, activation='relu'),
    #     Dense(num_classes, activation='softmax')
    # ])

    model = tf.keras.Sequential([
        tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False,
                                               weights='imagenet'),
        # tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / K.maximum(possible_positives, K.epsilon())
        recall = K.switch(K.equal(possible_positives, 0), 0.0, recall)
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / K.maximum(predicted_positives, K.epsilon())
        precision = K.switch(K.equal(predicted_positives, 0), 0.0, precision)
        return precision

    def f1(y_true, y_pred):
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
        f1_val = 2 * (prec * rec) / K.maximum(prec + rec, K.epsilon())
        return f1_val

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", precision, recall, f1])

    # Early stopping to prevent overtraining and to ensure decreasing validation loss
    # early_stop = EarlyStopping(monitor='val_loss',
    #                            patience=3,
    #                            restore_best_weights=True,
    #                            mode='min')
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=3,
                               restore_best_weights=True,
                               mode='min')

    # Train the model
    model.fit(train_images_split3, train_labels_split3_one_hot,
              validation_data=(val_images_split3, val_labels_split3_one_hot), batch_size=32, epochs=40,
              callbacks=[early_stop, PlotLossesKeras()], shuffle=True)

    # Evaluate the model on the test data
    score = model.evaluate(test_images_split3)
    print(score)

    # Get the predicted class probabilities for the test dataset
    y_pred_prob = model.predict(test_images_split3)

    # store the fpr, tpr, and roc_auc for all averaging strategies
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_labels_split3_one_hot.ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels_split3_one_hot[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(num_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    target_names = np.array(
        ['pilka_do_noznej', 'pilka_do_tenisa', 'pilka_do_koszykowki', 'pileczka_do_tenisa_stolowego',
         'pilka_do_siatkowski', 'pilka_do_recznej'])
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "limegreen", "purple", "gold"])
    for class_id, color in zip(range(num_classes), colors):
        RocCurveDisplay.from_predictions(
            test_labels_split3_one_hot[:, class_id],
            y_pred_prob[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.show()

    # Convert the predicted class probabilities to class labels
    y_pred = np.argmax(y_pred_prob, axis=1)

    y_true = np.argmax(test_labels_split3_one_hot, axis=1)
    conf_matr = confusion_matrix(y_true, y_pred)
    print("Confusion matrix: ")
    print(conf_matr)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=1)
    print("Recall:", recall)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=1)
    print("Precision:", precision)
    f1 = 2 * (precision * recall) / (precision + recall)
    print("F1 score: ", f1)

    counter = 0
    sum = 0
    for y in y_pred:
        for key, value in label_map.items():
            if value == y:
                if key == test_labels_split3[counter]:
                    sum += 1
        counter += 1
    accuracy = sum / len(y_pred)
    print(f'Accuracy = {accuracy}')


main()

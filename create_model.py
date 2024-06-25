"""
Build and train neural network models and output C header files for Embedded.

SPDX-License-Identifier: Apache-2.0
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import random
import tensorflow as tf
import training_data as data


def augment_data(data, sequence_length=20, num_augments=10):
    all_data = np.zeros((data.shape[0] * (num_augments + 1), sequence_length), dtype=int)
    all_data[:data.shape[0], :] = data

    for n in range(num_augments):
        for i, row in enumerate(data):
            active_indices = np.where(row == 1)[0]
            if len(active_indices) == 0:
                continue

            first_active = active_indices[0]
            last_active = active_indices[-1]

            max_shift_left = first_active
            max_shift_right = sequence_length - (last_active - first_active + 1)

            if max_shift_left + max_shift_right > 0:
                offset = random.randint(-max_shift_left, max_shift_right)
            else:
                offset = 0

            augmented_row = np.zeros(sequence_length, dtype=int)
            shift_start = max(0, first_active + offset)
            shift_end = min(sequence_length, last_active + offset + 1)
            length_to_copy = min(shift_end - shift_start, last_active - first_active + 1)
            augmented_row[shift_start:shift_start + length_to_copy] = row[first_active:first_active + length_to_copy]

            all_data[data.shape[0] + n * data.shape[0] + i, :] = augmented_row
    return all_data

def convert_to_c_array(bytes_data):
    hex_array = [format(x, '#04x') for x in bytes_data]
    c_array = ''
    for i, hex_val in enumerate(hex_array):
        if i % 10 == 0 and i != 0:
            c_array += '\n'
        c_array += hex_val + ', '
    return c_array.strip().rstrip(',')

def main():
    print("Augment the training data set");
    # augument training dataset
    nop = augment_data(data.nop, num_augments=100)
    single_click = augment_data(data.single_click, num_augments=100)
    double_click = augment_data(data.double_click, num_augments=100)

    print("Assign labels to data sets");
    all_data = np.concatenate((nop, single_click, double_click), axis=0)
    # create label
    labels_nop = np.full(len(nop), 0)
    labels_single_click = np.full(len(single_click), 1)
    labels_double_click = np.full(len(double_click), 2)
    all_labels = np.concatenate((labels_nop, labels_single_click, labels_double_click), axis=0)

    print("Split the dataset for training and testing");
    X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)

    print("Create and train a model")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(20,)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

    print("Evaluate trained models with test data")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)

    print("Convert to model.tflite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)

    #with open('model_quantized.tflite', 'rb') as f:
    #    tflite_model = f.read()

    print("Convert model to model.h file")
    c_array = convert_to_c_array(tflite_model)
    header_file_content = f"""
#ifndef MODEL_H
#define MODEL_H

const unsigned char model_tflite[] = {{
{c_array}
}};

const int model_tflite_len = {len(tflite_model)};

#endif  // MODEL_H
"""
    with open('model.h', 'w') as f:
        f.write(header_file_content)


if __name__ == '__main__':
    main();

import os
import random

import numpy as np
import tensorflow as tf


def check_data():
    choices = {"no_attacks": no_attacks,
               "attack_closest_to_nexus": attack_closest_to_nexus,
               "attack_enemy_structures": attack_enemy_structures,
               "attack_enemy_start": attack_enemy_start,
               "attack_enemy_target": attack_enemy_target}

    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:", total_data)
    return lengths


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                 input_shape=(168, 192, 3),
                 activation='relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(6, activation='softmax'))

learning_rate = 0.0001
opt = tf.keras.optimizers.Adam(learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/stage1")

train_data_dir = "train_data"

hm_epochs = 10

for i in range(hm_epochs):
    current = 0
    increment = 1383
    not_maximum = True
    all_files = os.listdir(train_data_dir)
    maximum = len(all_files)
    random.shuffle(all_files)
    print(f"WORKING ON {current}:{current + increment}")
    print(f"MAXIMUM = {maximum}")

    no_attacks = []
    attack_closest_to_nexus = []
    attack_enemy_structures = []
    attack_enemy_start = []
    attack_enemy_target = []
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    for file in all_files[current:current + increment]:
        full_path = os.path.join(train_data_dir, file)
        data = np.load(full_path)
        data = list(data)
        for d in data:
            choice = np.argmax(d[0])
            if choice == 0 or choice == 5 or choice == 6:
                no_attacks.append([d[0], d[1]])
            elif choice == 1:
                attack_closest_to_nexus.append([d[0], d[1]])
            elif choice == 2:
                attack_enemy_structures.append([d[0], d[1]])
            elif choice == 3:
                attack_enemy_start.append([d[0], d[1]])
            elif choice == 4:
                attack_enemy_target.append([d[0], d[1]])
    lengths = check_data()
    lowest_data = min(lengths)

    random.shuffle(no_attacks)
    random.shuffle(attack_closest_to_nexus)
    random.shuffle(attack_enemy_structures)
    random.shuffle(attack_enemy_start)

    no_attacks = no_attacks[:lowest_data]
    attack_closest_to_nexus = attack_closest_to_nexus[:lowest_data]
    attack_enemy_structures = attack_enemy_structures[:lowest_data]
    attack_enemy_start = attack_enemy_start[:lowest_data]

    check_data()
    train_data = no_attacks + attack_closest_to_nexus + attack_enemy_structures + attack_enemy_start + attack_enemy_target
    random.shuffle(train_data)
    print(len(train_data))
    test_size = 100
    batch_size = 128
    x_train = np.array([i[1] for i in (train_data[:-test_size])], np.uint8)
    y_train = np.array([i[0] for i in (train_data[:-test_size])])

    x_test = np.array([i[1] for i in (train_data[-test_size:])])
    y_test = np.array([i[0] for i in (train_data[-test_size:])])

    print(x_train.shape)
    print(x_test.shape)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data=(x_test, y_test),
              shuffle=True,
              verbose=1, callbacks=[tensorboard])
    model.save("BasicCNN-{}-epochs-{}-LR-STAGE1".format(hm_epochs, learning_rate))
    current += increment

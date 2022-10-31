#!/usr/bin/env python
# coding: utf-8

# # 3D-Mel spectrogram+MFCCs+Chromagram VGG-opti-VMD EMoDB
# 
# ### Author @ David Hason

# ### Import required libraries

## Modelling

from tqdm import tqdm
import librosa as lb
from vmdpy import VMD  

def energy(u):
    # Estimate PSD `S_xx_welch` at discrete frequencies `f_welch`
    f_welch, S_xx_welch = scipy.signal.welch(u)
    # Integrate PSD over spectral bandwidth
    # to obtain signal power `P_welch`
    df_welch = f_welch[1] - f_welch[0]
    return np.sum(S_xx_welch) * df_welch  

def maxvdm(f, alpha, K):
    tau = 0            
    DC = 0             
    init = 1           
    tol = 1e-9
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol) 
    energy_array=[]
    for i in u:
        energy_array.append(energy(i))
    ind = np.argmax(energy_array)
    return u[ind]

def vm(features, alpha, K):
    X = []
    for i in tqdm(features):
        X.append(maxvdm(i, alpha, K))
    return X

from tensorflow.keras.applications import VGG16
import tensorflow as tf
input_shape = (X_train.shape[1], X_train.shape[2],  X_train.shape[3])

VGG16_MODEL = VGG16(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')
VGG16_MODEL.trainable=False
flatten_layer = tf.keras.layers.Flatten()
global_average_layer = tf.keras.layers.MaxPool2D()
prediction_layer1 = tf.keras.layers.Dense(4096,activation='tanh')
prediction_layer2 = tf.keras.layers.Dense(2048,activation='selu')
prediction_layer3 = tf.keras.layers.Dense(1024,activation='selu')
prediction_layer4 = tf.keras.layers.Dense(512,activation='relu')
prediction_layer5 = tf.keras.layers.Dense(256,activation='relu')
prediction_layer6 = tf.keras.layers.Dense(128,activation='relu')
prediction_layer7 = tf.keras.layers.Dense(7,activation='softmax')
model = tf.keras.Sequential([
  VGG16_MODEL,
  global_average_layer,
  flatten_layer,
  prediction_layer1,
  prediction_layer2,
  prediction_layer3,
  prediction_layer4,
  prediction_layer5,
  prediction_layer6,
  prediction_layer7
])
saved_model="vgg16_mfcc_13_2048_512.weights.hdf5"
optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
checkpoint = ModelCheckpoint(saved_model, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.summary()

## alpha = 2000
## K = 4

alpha = [2000]
K = [4]

for alpha_val, K_val in zip(alpha, K):
    print('================')
    print('Value of K = ', K_val)
    print('==============')
    print('Value of alpha = ', alpha_val)
    print('=======================')
    
    X_train_vmd = vm(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3]), alpha = alpha_val, K = K_val)
    X_test_vmd = vm(X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3]), alpha = alpha_val, K = K_val)
    X_train_vmd = np.stack(X_train_vmd, axis=0)
    X_test_vmd = np.stack(X_test_vmd, axis=0)

    X_train_vmd = np.reshape(X_train_vmd, (X_train_vmd.shape[0], 128,128,3))
    X_test_vmd = np.reshape(X_test_vmd, (X_test_vmd.shape[0], 128,128,3))

    print('Shape of Training Features : ', X_train_vmd.shape)
    print('Shape of Testing Features : ', X_test_vmd.shape)
    print('Shape of Training labels : ', y_train.shape)
    print('Shape of Testing labels : ', y_test.shape)

    unique_elements, counts_elements = np.unique(np.argmax(y_train,axis = 1), return_counts=True)
    print("Frequency of unique values of the Training array:")
    print(np.asarray((unique_elements, counts_elements)))

    unique_elements, counts_elements = np.unique(np.argmax(y_test,axis = 1), return_counts=True)
    print("Frequency of unique values of the Testing array:")
    print(np.asarray((unique_elements, counts_elements)))

    import tensorflow as tf
    def setup_gpus():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0],'GPU')
                tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
            except RuntimeError as e:
                print(e)

    print('=====================')
    print('Model Training ......')
    history = model.fit(X_train_vmd, y_train, validation_data = (X_test_vmd, y_test), batch_size=4, epochs=50, 
                    verbose=1)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    t = f.suptitle('VGG16 Neural Net Performance (1D-MFCC)', fontsize=18, color = 'red')
    f.subplots_adjust(top=0.85, wspace=0.2)
    epochs = list(range(1,51))
    ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], label='Test Accuracy')
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epochs, history.history['loss'], label='Train Loss')
    ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
    plt.show();

    print('=====================')
    print('Evaluating Model ....')
    print('=====================')

    predictions = np.argmax(model.predict(X_test_vmd), axis=-1)

    class_map = {'0' : 'anger', '1' : 'boredom', '2' : 'disgust', '3' : 'fear', '4' : 'happiness', 
                 '5' : 'neutral', '6' : 'sadness'}

    test_labels_categories = [class_map[str(label)] for label in np.argmax(y_test,axis = 1)]
    prediction_labels_categories = [class_map[str(label)] for label in predictions]
    category_names = list(class_map.values())

    from sklearn.exceptions import UndefinedMetricWarning
    print('===============')
    print('Overall Metrics')
    print('===============')
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    get_metrics(true_labels=test_labels_categories, 
                   predicted_labels=prediction_labels_categories)

    print('=====================================')
    print('Classification Performance Class-Wise')
    print('=====================================')

    display_classification_report(true_labels=test_labels_categories, 
                                        predicted_labels=prediction_labels_categories, 
                                        classes=category_names)
    def classwise_accuracy():
        a = pd.crosstab(np.array(test_labels_categories) , np.array(prediction_labels_categories))
        return a.max(axis=1)/a.sum(axis=1)
    accuracy_per_class = classwise_accuracy()
    print(accuracy_per_class)

    print('============================')
    print('Normalised Confusion Matrix ')
    print('============================')
    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(test_labels_categories, prediction_labels_categories)
    pd.options.display.float_format = "{:,.2f}".format

    CF = pd.DataFrame(C / C.astype(np.float).sum(axis=1), columns=['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness'])*100
    CF.index = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness']
    display(CF)
    import seaborn as sns
    sns.heatmap(confusion_matrix(test_labels_categories, prediction_labels_categories), annot= True)
    plt.show();

## alpha = 3000
## K = 4

alpha = [3000]
K = [4]

for alpha_val, K_val in zip(alpha, K):
    print('================')
    print('Value of K = ', K_val)
    print('==============')
    print('Value of alpha = ', alpha_val)
    print('=======================')
    
    X_train_vmd = vm(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3]), alpha = alpha_val, K = K_val)
    X_test_vmd = vm(X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3]), alpha = alpha_val, K = K_val)
    X_train_vmd = np.stack(X_train_vmd, axis=0)
    X_test_vmd = np.stack(X_test_vmd, axis=0)

    X_train_vmd = np.reshape(X_train_vmd, (X_train_vmd.shape[0], 128,128,3))
    X_test_vmd = np.reshape(X_test_vmd, (X_test_vmd.shape[0], 128,128,3))

    print('Shape of Training Features : ', X_train_vmd.shape)
    print('Shape of Testing Features : ', X_test_vmd.shape)
    print('Shape of Training labels : ', y_train.shape)
    print('Shape of Testing labels : ', y_test.shape)

    unique_elements, counts_elements = np.unique(np.argmax(y_train,axis = 1), return_counts=True)
    print("Frequency of unique values of the Training array:")
    print(np.asarray((unique_elements, counts_elements)))

    unique_elements, counts_elements = np.unique(np.argmax(y_test,axis = 1), return_counts=True)
    print("Frequency of unique values of the Testing array:")
    print(np.asarray((unique_elements, counts_elements)))

    import tensorflow as tf
    def setup_gpus():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0],'GPU')
                tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
            except RuntimeError as e:
                print(e)

    print('=====================')
    print('Model Training ......')
    history = model.fit(X_train_vmd, y_train, validation_data = (X_test_vmd, y_test), batch_size=4, epochs=50, 
                    verbose=1)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    t = f.suptitle('VGG16 Neural Net Performance (1D-MFCC)', fontsize=18, color = 'red')
    f.subplots_adjust(top=0.85, wspace=0.2)
    epochs = list(range(1,51))
    ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], label='Test Accuracy')
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epochs, history.history['loss'], label='Train Loss')
    ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
    plt.show();

    print('=====================')
    print('Evaluating Model ....')
    print('=====================')

    predictions = np.argmax(model.predict(X_test_vmd), axis=-1)

    class_map = {'0' : 'anger', '1' : 'boredom', '2' : 'disgust', '3' : 'fear', '4' : 'happiness', 
                 '5' : 'neutral', '6' : 'sadness'}

    test_labels_categories = [class_map[str(label)] for label in np.argmax(y_test,axis = 1)]
    prediction_labels_categories = [class_map[str(label)] for label in predictions]
    category_names = list(class_map.values())

    from sklearn.exceptions import UndefinedMetricWarning
    print('===============')
    print('Overall Metrics')
    print('===============')
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    get_metrics(true_labels=test_labels_categories, 
                   predicted_labels=prediction_labels_categories)

    print('=====================================')
    print('Classification Performance Class-Wise')
    print('=====================================')

    display_classification_report(true_labels=test_labels_categories, 
                                        predicted_labels=prediction_labels_categories, 
                                        classes=category_names)
    def classwise_accuracy():
        a = pd.crosstab(np.array(test_labels_categories) , np.array(prediction_labels_categories))
        return a.max(axis=1)/a.sum(axis=1)
    accuracy_per_class = classwise_accuracy()
    print(accuracy_per_class)

    print('============================')
    print('Normalised Confusion Matrix ')
    print('============================')
    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(test_labels_categories, prediction_labels_categories)
    pd.options.display.float_format = "{:,.2f}".format

    CF = pd.DataFrame(C / C.astype(np.float).sum(axis=1), columns=['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness'])*100
    CF.index = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness']
    display(CF)
    import seaborn as sns
    sns.heatmap(confusion_matrix(test_labels_categories, prediction_labels_categories), annot= True)
    plt.show();

## alpha = 4000
## K = 4

alpha = [4000]
K = [4]

for alpha_val, K_val in zip(alpha, K):
    print('================')
    print('Value of K = ', K_val)
    print('==============')
    print('Value of alpha = ', alpha_val)
    print('=======================')
    
    X_train_vmd = vm(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3]), alpha = alpha_val, K = K_val)
    X_test_vmd = vm(X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3]), alpha = alpha_val, K = K_val)
    X_train_vmd = np.stack(X_train_vmd, axis=0)
    X_test_vmd = np.stack(X_test_vmd, axis=0)

    X_train_vmd = np.reshape(X_train_vmd, (X_train_vmd.shape[0], 128,128,3))
    X_test_vmd = np.reshape(X_test_vmd, (X_test_vmd.shape[0], 128,128,3))

    print('Shape of Training Features : ', X_train_vmd.shape)
    print('Shape of Testing Features : ', X_test_vmd.shape)
    print('Shape of Training labels : ', y_train.shape)
    print('Shape of Testing labels : ', y_test.shape)

    unique_elements, counts_elements = np.unique(np.argmax(y_train,axis = 1), return_counts=True)
    print("Frequency of unique values of the Training array:")
    print(np.asarray((unique_elements, counts_elements)))

    unique_elements, counts_elements = np.unique(np.argmax(y_test,axis = 1), return_counts=True)
    print("Frequency of unique values of the Testing array:")
    print(np.asarray((unique_elements, counts_elements)))

    import tensorflow as tf
    def setup_gpus():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0],'GPU')
                tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
            except RuntimeError as e:
                print(e)

    print('=====================')
    print('Model Training ......')
    history = model.fit(X_train_vmd, y_train, validation_data = (X_test_vmd, y_test), batch_size=4, epochs=50, 
                    verbose=1)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    t = f.suptitle('VGG16 Neural Net Performance (1D-MFCC)', fontsize=18, color = 'red')
    f.subplots_adjust(top=0.85, wspace=0.2)
    epochs = list(range(1,51))
    ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], label='Test Accuracy')
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epochs, history.history['loss'], label='Train Loss')
    ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
    plt.show();

    print('=====================')
    print('Evaluating Model ....')
    print('=====================')

    predictions = np.argmax(model.predict(X_test_vmd), axis=-1)

    class_map = {'0' : 'anger', '1' : 'boredom', '2' : 'disgust', '3' : 'fear', '4' : 'happiness', 
                 '5' : 'neutral', '6' : 'sadness'}

    test_labels_categories = [class_map[str(label)] for label in np.argmax(y_test,axis = 1)]
    prediction_labels_categories = [class_map[str(label)] for label in predictions]
    category_names = list(class_map.values())

    from sklearn.exceptions import UndefinedMetricWarning
    print('===============')
    print('Overall Metrics')
    print('===============')
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    get_metrics(true_labels=test_labels_categories, 
                   predicted_labels=prediction_labels_categories)

    print('=====================================')
    print('Classification Performance Class-Wise')
    print('=====================================')

    display_classification_report(true_labels=test_labels_categories, 
                                        predicted_labels=prediction_labels_categories, 
                                        classes=category_names)
    def classwise_accuracy():
        a = pd.crosstab(np.array(test_labels_categories) , np.array(prediction_labels_categories))
        return a.max(axis=1)/a.sum(axis=1)
    accuracy_per_class = classwise_accuracy()
    print(accuracy_per_class)

    print('============================')
    print('Normalised Confusion Matrix ')
    print('============================')
    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(test_labels_categories, prediction_labels_categories)
    pd.options.display.float_format = "{:,.2f}".format

    CF = pd.DataFrame(C / C.astype(np.float).sum(axis=1), columns=['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness'])*100
    CF.index = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness']
    display(CF)
    import seaborn as sns
    sns.heatmap(confusion_matrix(test_labels_categories, prediction_labels_categories), annot= True)
    plt.show();

## alpha = 2000
## K = 6

alpha = [2000]
K = [6]

for alpha_val, K_val in zip(alpha, K):
    print('================')
    print('Value of K = ', K_val)
    print('==============')
    print('Value of alpha = ', alpha_val)
    print('=======================')
    
    X_train_vmd = vm(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3]), alpha = alpha_val, K = K_val)
    X_test_vmd = vm(X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3]), alpha = alpha_val, K = K_val)
    X_train_vmd = np.stack(X_train_vmd, axis=0)
    X_test_vmd = np.stack(X_test_vmd, axis=0)

    X_train_vmd = np.reshape(X_train_vmd, (X_train_vmd.shape[0], 128,128,3))
    X_test_vmd = np.reshape(X_test_vmd, (X_test_vmd.shape[0], 128,128,3))

    print('Shape of Training Features : ', X_train_vmd.shape)
    print('Shape of Testing Features : ', X_test_vmd.shape)
    print('Shape of Training labels : ', y_train.shape)
    print('Shape of Testing labels : ', y_test.shape)

    unique_elements, counts_elements = np.unique(np.argmax(y_train,axis = 1), return_counts=True)
    print("Frequency of unique values of the Training array:")
    print(np.asarray((unique_elements, counts_elements)))

    unique_elements, counts_elements = np.unique(np.argmax(y_test,axis = 1), return_counts=True)
    print("Frequency of unique values of the Testing array:")
    print(np.asarray((unique_elements, counts_elements)))

    import tensorflow as tf
    def setup_gpus():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0],'GPU')
                tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
            except RuntimeError as e:
                print(e)

    print('=====================')
    print('Model Training ......')
    history = model.fit(X_train_vmd, y_train, validation_data = (X_test_vmd, y_test), batch_size=4, epochs=50, 
                    verbose=1)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    t = f.suptitle('VGG16 Neural Net Performance (1D-MFCC)', fontsize=18, color = 'red')
    f.subplots_adjust(top=0.85, wspace=0.2)
    epochs = list(range(1,51))
    ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], label='Test Accuracy')
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epochs, history.history['loss'], label='Train Loss')
    ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
    plt.show();

    print('=====================')
    print('Evaluating Model ....')
    print('=====================')

    predictions = np.argmax(model.predict(X_test_vmd), axis=-1)

    class_map = {'0' : 'anger', '1' : 'boredom', '2' : 'disgust', '3' : 'fear', '4' : 'happiness', 
                 '5' : 'neutral', '6' : 'sadness'}

    test_labels_categories = [class_map[str(label)] for label in np.argmax(y_test,axis = 1)]
    prediction_labels_categories = [class_map[str(label)] for label in predictions]
    category_names = list(class_map.values())

    from sklearn.exceptions import UndefinedMetricWarning
    print('===============')
    print('Overall Metrics')
    print('===============')
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    get_metrics(true_labels=test_labels_categories, 
                   predicted_labels=prediction_labels_categories)

    print('=====================================')
    print('Classification Performance Class-Wise')
    print('=====================================')

    display_classification_report(true_labels=test_labels_categories, 
                                        predicted_labels=prediction_labels_categories, 
                                        classes=category_names)
    def classwise_accuracy():
        a = pd.crosstab(np.array(test_labels_categories) , np.array(prediction_labels_categories))
        return a.max(axis=1)/a.sum(axis=1)
    accuracy_per_class = classwise_accuracy()
    print(accuracy_per_class)

    print('============================')
    print('Normalised Confusion Matrix ')
    print('============================')
    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(test_labels_categories, prediction_labels_categories)
    pd.options.display.float_format = "{:,.2f}".format

    CF = pd.DataFrame(C / C.astype(np.float).sum(axis=1), columns=['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness'])*100
    CF.index = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness']
    display(CF)
    import seaborn as sns
    sns.heatmap(confusion_matrix(test_labels_categories, prediction_labels_categories), annot= True)
    plt.show();

## alpha = 3000
## K = 6

alpha = [3000]
K = [6]

for alpha_val, K_val in zip(alpha, K):
    print('================')
    print('Value of K = ', K_val)
    print('==============')
    print('Value of alpha = ', alpha_val)
    print('=======================')
    
    X_train_vmd = vm(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3]), alpha = alpha_val, K = K_val)
    X_test_vmd = vm(X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3]), alpha = alpha_val, K = K_val)
    X_train_vmd = np.stack(X_train_vmd, axis=0)
    X_test_vmd = np.stack(X_test_vmd, axis=0)

    X_train_vmd = np.reshape(X_train_vmd, (X_train_vmd.shape[0], 128,128,3))
    X_test_vmd = np.reshape(X_test_vmd, (X_test_vmd.shape[0], 128,128,3))

    print('Shape of Training Features : ', X_train_vmd.shape)
    print('Shape of Testing Features : ', X_test_vmd.shape)
    print('Shape of Training labels : ', y_train.shape)
    print('Shape of Testing labels : ', y_test.shape)

    unique_elements, counts_elements = np.unique(np.argmax(y_train,axis = 1), return_counts=True)
    print("Frequency of unique values of the Training array:")
    print(np.asarray((unique_elements, counts_elements)))

    unique_elements, counts_elements = np.unique(np.argmax(y_test,axis = 1), return_counts=True)
    print("Frequency of unique values of the Testing array:")
    print(np.asarray((unique_elements, counts_elements)))

    import tensorflow as tf
    def setup_gpus():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0],'GPU')
                tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
            except RuntimeError as e:
                print(e)

    print('=====================')
    print('Model Training ......')
    history = model.fit(X_train_vmd, y_train, validation_data = (X_test_vmd, y_test), batch_size=4, epochs=50, 
                    verbose=1)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    t = f.suptitle('VGG16 Neural Net Performance (1D-MFCC)', fontsize=18, color = 'red')
    f.subplots_adjust(top=0.85, wspace=0.2)
    epochs = list(range(1,51))
    ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], label='Test Accuracy')
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epochs, history.history['loss'], label='Train Loss')
    ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
    plt.show();

    print('=====================')
    print('Evaluating Model ....')
    print('=====================')

    predictions = np.argmax(model.predict(X_test_vmd), axis=-1)

    class_map = {'0' : 'anger', '1' : 'boredom', '2' : 'disgust', '3' : 'fear', '4' : 'happiness', 
                 '5' : 'neutral', '6' : 'sadness'}

    test_labels_categories = [class_map[str(label)] for label in np.argmax(y_test,axis = 1)]
    prediction_labels_categories = [class_map[str(label)] for label in predictions]
    category_names = list(class_map.values())

    from sklearn.exceptions import UndefinedMetricWarning
    print('===============')
    print('Overall Metrics')
    print('===============')
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    get_metrics(true_labels=test_labels_categories, 
                   predicted_labels=prediction_labels_categories)

    print('=====================================')
    print('Classification Performance Class-Wise')
    print('=====================================')

    display_classification_report(true_labels=test_labels_categories, 
                                        predicted_labels=prediction_labels_categories, 
                                        classes=category_names)
    def classwise_accuracy():
        a = pd.crosstab(np.array(test_labels_categories) , np.array(prediction_labels_categories))
        return a.max(axis=1)/a.sum(axis=1)
    accuracy_per_class = classwise_accuracy()
    print(accuracy_per_class)

    print('============================')
    print('Normalised Confusion Matrix ')
    print('============================')
    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(test_labels_categories, prediction_labels_categories)
    pd.options.display.float_format = "{:,.2f}".format

    CF = pd.DataFrame(C / C.astype(np.float).sum(axis=1), columns=['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness'])*100
    CF.index = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness']
    display(CF)
    import seaborn as sns
    sns.heatmap(confusion_matrix(test_labels_categories, prediction_labels_categories), annot= True)
    plt.show();

## alpha = 4000
## K = 6

alpha = [4000]
K = [6]

for alpha_val, K_val in zip(alpha, K):
    print('================')
    print('Value of K = ', K_val)
    print('==============')
    print('Value of alpha = ', alpha_val)
    print('=======================')
    
    X_train_vmd = vm(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3]), alpha = alpha_val, K = K_val)
    X_test_vmd = vm(X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]*X_test.shape[3]), alpha = alpha_val, K = K_val)
    X_train_vmd = np.stack(X_train_vmd, axis=0)
    X_test_vmd = np.stack(X_test_vmd, axis=0)

    X_train_vmd = np.reshape(X_train_vmd, (X_train_vmd.shape[0], 128,128,3))
    X_test_vmd = np.reshape(X_test_vmd, (X_test_vmd.shape[0], 128,128,3))

    print('Shape of Training Features : ', X_train_vmd.shape)
    print('Shape of Testing Features : ', X_test_vmd.shape)
    print('Shape of Training labels : ', y_train.shape)
    print('Shape of Testing labels : ', y_test.shape)

    unique_elements, counts_elements = np.unique(np.argmax(y_train,axis = 1), return_counts=True)
    print("Frequency of unique values of the Training array:")
    print(np.asarray((unique_elements, counts_elements)))

    unique_elements, counts_elements = np.unique(np.argmax(y_test,axis = 1), return_counts=True)
    print("Frequency of unique values of the Testing array:")
    print(np.asarray((unique_elements, counts_elements)))

    import tensorflow as tf
    def setup_gpus():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0],'GPU')
                tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
            except RuntimeError as e:
                print(e)

    print('=====================')
    print('Model Training ......')
    history = model.fit(X_train_vmd, y_train, validation_data = (X_test_vmd, y_test), batch_size=4, epochs=50, 
                    verbose=1)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    t = f.suptitle('VGG16 Neural Net Performance (1D-MFCC)', fontsize=18, color = 'red')
    f.subplots_adjust(top=0.85, wspace=0.2)
    epochs = list(range(1,51))
    ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epochs, history.history['val_accuracy'], label='Test Accuracy')
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epochs, history.history['loss'], label='Train Loss')
    ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
    plt.show();

    print('=====================')
    print('Evaluating Model ....')
    print('=====================')

    predictions = np.argmax(model.predict(X_test_vmd), axis=-1)

    class_map = {'0' : 'anger', '1' : 'boredom', '2' : 'disgust', '3' : 'fear', '4' : 'happiness', 
                 '5' : 'neutral', '6' : 'sadness'}

    test_labels_categories = [class_map[str(label)] for label in np.argmax(y_test,axis = 1)]
    prediction_labels_categories = [class_map[str(label)] for label in predictions]
    category_names = list(class_map.values())

    from sklearn.exceptions import UndefinedMetricWarning
    print('===============')
    print('Overall Metrics')
    print('===============')
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    get_metrics(true_labels=test_labels_categories, 
                   predicted_labels=prediction_labels_categories)

    print('=====================================')
    print('Classification Performance Class-Wise')
    print('=====================================')

    display_classification_report(true_labels=test_labels_categories, 
                                        predicted_labels=prediction_labels_categories, 
                                        classes=category_names)
    def classwise_accuracy():
        a = pd.crosstab(np.array(test_labels_categories) , np.array(prediction_labels_categories))
        return a.max(axis=1)/a.sum(axis=1)
    accuracy_per_class = classwise_accuracy()
    print(accuracy_per_class)

    print('============================')
    print('Normalised Confusion Matrix ')
    print('============================')
    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(test_labels_categories, prediction_labels_categories)
    pd.options.display.float_format = "{:,.2f}".format

    CF = pd.DataFrame(C / C.astype(np.float).sum(axis=1), columns=['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness'])*100
    CF.index = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness']
    display(CF)
    import seaborn as sns
    sns.heatmap(confusion_matrix(test_labels_categories, prediction_labels_categories), annot= True)
    plt.show();




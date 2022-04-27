import os
import pandas
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import itertools

image_height = 224
image_width = 224
batch_size = 64
image_catalog = 4
epochs = 30

PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT,"covid_dataset_3")
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH,"train/train")
VALIDATION_DATASET_PATH = os.path.join(DATASET_PATH,"validation/validation")
TEST_DATASET_PATH = os.path.join(DATASET_PATH,"test/test")

train_data_configuration = ImageDataGenerator(rescale = 1./255, rotation_range=45, height_shift_range=0.2, width_shift_range=0.2, shear_range = 0.2, zoom_range=0.2,
                                                horizontal_flip = True, fill_mode ='nearest')

train_dataset = train_data_configuration.flow_from_directory(directory=TRAIN_DATASET_PATH, target_size=(image_height,image_width), batch_size=batch_size, class_mode='categorical', shuffle=True)

validation_data_configuration = ImageDataGenerator(rescale=1./255)
validation_dataset = validation_data_configuration.flow_from_directory(directory=VALIDATION_DATASET_PATH, target_size=(image_height,image_width), batch_size =batch_size, class_mode='categorical', shuffle=False)

test_data_configuration = ImageDataGenerator(rescale=1./255)
test_dataset = test_data_configuration.flow_from_directory(directory=TEST_DATASET_PATH, target_size=(image_height,image_width), batch_size =batch_size, class_mode='categorical', shuffle=False)

print(train_dataset.n)
print(validation_dataset.n)
print(test_dataset.n)

train_steps_per_epoch = train_dataset.samples//batch_size

validation_steps_per_epoch = validation_dataset.samples//batch_size
test_steps_per_epoch = test_dataset.samples//batch_size

CNN_base_inception_v3 = tf.keras.applications.InceptionV3(weights='imagenet',include_top=False,input_shape=(224,224,3))
#CNN_base_inception_v3 = tf.keras.applications.InceptionResNetV2(weights='imagenet',include_top=False,input_shape=(224,224,3))
#CNN_base_inception_v3 = tf.keras.applications.VGG19(weights='imagenet',include_top=False,input_shape=(224,224,3))

model = tf.keras.Sequential()
model.add(CNN_base_inception_v3)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(4,activation='softmax'))
model.summary()

for layer in CNN_base_inception_v3.layers:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
model_history_tl = model.fit(train_dataset, steps_per_epoch=train_steps_per_epoch, epochs= epochs - 20, validation_data=validation_dataset,
                          validation_steps=validation_steps_per_epoch)

for layer in CNN_base_inception_v3.layers[:20]:
        layer.trainable = False
for layer in CNN_base_inception_v3.layers[20:]:
        layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=["accuracy"])
model_history_ft = model.fit(train_dataset, steps_per_epoch=train_steps_per_epoch, epochs= epochs, validation_data=validation_dataset,
                          validation_steps=validation_steps_per_epoch)

train_accuracy = model_history_ft.history['accuracy']
validation_accuracy = model_history_ft.history['val_accuracy']
train_loss = model_history_ft.history['loss']
validation_loss = model_history_ft.history['val_loss']

epochs_values = []
for i in range(1, epochs + 1):
    epochs_values.append(i)

plt.figure(figsize=(10,5))
plt.plot(epochs_values, train_accuracy, label='Train Accuracy')
plt.plot(epochs_values, validation_accuracy, label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs Values')
plt.title('The Train and Validation Accuracy')
plt.savefig('train_validation_accuracy.jpg')
#plt.show()

plt.figure(figsize=(10,5))
plt.plot(epochs_values, train_loss, label='Train Loss')
plt.plot(epochs_values, validation_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epochs Values')
plt.title('The Train and Validation Loss')
plt.savefig('train_validation_loss.jpg')
#plt.show()

test_prediction = model.predict_generator(test_dataset,test_steps_per_epoch+1)
test_prediction_classes = np.argmax(test_prediction,axis=1)
confusion_matrix_result =confusion_matrix(test_dataset.classes, test_prediction_classes)
#print("The confusion matrix is:",confusion_matrix_result)

def confusion_matrix(confusion_matrix, label_names):

    accuracy_rate = np.trace(confusion_matrix)/float(np.sum(confusion_matrix))
    error_rate = 1 - accuracy_rate

    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion matrix')
    plt.colorbar()

    tick = np.arange(len(label_names))
    plt.xticks(tick, label_names, rotation=45)
    plt.yticks(tick, label_names)

    confusion_mat = np.round(confusion_matrix.astype('float32') / confusion_matrix.sum(axis=1), 2)

    threshold = confusion_mat.max() / 1.5
    matrix_product = itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1]))
    for i, j in matrix_product:
        plt.text(j, i, "{:0.2f}".format(confusion_mat[i, j]), horizontalalignment="center", color="white" if confusion_mat[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('Target class')
    plt.xlabel("Output class\n accuracy rate = {:0.2f}\n error rate = {:0.2f}".format(accuracy_rate, error_rate))
    plt.savefig('confusion_matrix.jpg')
    #plt.show()

labels = ["covid", "lung_opacity", "normal", "viral_pneumonia"]
confusion_matrix(confusion_matrix_result, label_names=labels)

report = classification_report(test_dataset.classes, test_prediction_classes, target_names=labels, output_dict=True)
report_df = pandas.DataFrame(report).transpose()
print(report)
report_df.to_csv('classification_report.csv', index= True)

# Garbage_Classification_CNN


# Evrişimli Sinir Ağları Uygulama (Convolutional Neural Network Exercises)


## Google Colab - Google Drive Bağlantısının Yapılması

```python
from google.colab import drive
drive.mount('/content/drive')
```


# Evrişimli Sinir Ağları (CNN) ile Katı Atık Tespiti

1) İş Problemi (Business Problem)

2) Veriyi Anlamak (Data Understanding)

3) Veriyi Hazırlamak (Data Preparation)

4) Modelleme (Modeling)

5) Değerlendirme (Evaluation)



# 1) İş Problemi (Business Problem)

# 2) Veriyi Anlamak (Data Understanding)

## Veri Seti Hikayesi (Dataset Story)

Bu projed kapsamında kullanacağımız veri seti TrashNet isimli veri setidir. Stanaford Üniversitesi öğrencileri tarafından hazırlanmıştır. Veri seti altı farklı sınıftan oluşmaktadır. Veri setinde Cam, Kağıt, Karton, Plastik, Metal ve Çöp olmak üzere toplamda 2527 adet görüntü bulunmaktadır.


Görüntülerin dağılımı:

- 501 cam (glasses)
- 594 kağıt (paper)
- 403 karton (cardboard)
- 482 platik (plastic)
- 410 metal (metal)
- 137 çöp (trash)


Görüntüler beyaz bir panoya yerleştirilerek ve güneş ışığı veya oda aydınlatması kullanılarak çekilmiştir. Görüntüler, 512x384 piksel boyutlarında ve 3(RGB) kanallıdır.
- Veri seti hakkında daha fazla bilgi için GitHub sayasını ziyaret edebilirsiniz.
- Veri seti indirmek için Kaggle sayfasını ziyaret edebilirsiniz.


## 2.1) Gerekli  Kütüphanelerin Import İşlemleri

```python
# pip install imutils

#  Veriyi okuma ve işleme adımında kullanılacak olan kütüphaneler
import cv2
import urllib
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import random, os, glob
from imutils import paths
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from urllib.request import urlopen
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Warningleri kapatmak için kullanılmaktadır.
import warnings
warnings.filterwarnings('ignore')

#Model için kullanılacak olan kütüphaneler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
```


## 2.2) Veri Setinin Okunması

1.   Google Colab Notebook ile Google Drive hesabının eşleştirilmesi.
2.   Veri setinin Google Drive'a yüklenmesi ve adresin notebook'a gösterilmesi
3.   Veri setini okuyacak fonksiyonun tanımlanması.

```python
# from google.colab import drive
# drive.mount('/content/drive')
```
```python
# Drive'da bu veri setinde yer alan görüntülerin olduğu path bilgisinin tutulması
dir_path = '/content/drive/MyDrive/Garbage_classification'
```
```python
# Target size ve Label Etiket Değerlerinin Belirlenmesi

target_size = (224, 224)

waste_labels = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
```


## 2.3) Veri Setinden Örnekler Gösterilmesi

```python
def load_datasets(path):
  """
  Görsellerin bulunduğu dizindeki görüntüyü okuyup etiketleri oluşturur.

  Parametreler:

  path: Görsellerin bulunduğu dizini ifade eder.

  Return:
  
  x: Görüntülere ait olduğu sınıf bilgisini tutan liste.

  """
  
  x = []
  labels = []

  # Gönderdiğimiz pathdeki görüntüleri listeleyip sıralamaktadır.
  image_paths = sorted(list(paths.list_images(path)))

  for image_path in image_paths:
    # Belirtilen pathdeki görüntüler openCV kütüphanesi ile okunmaktadır.
    img = cv2.imread(image_path)

    # Okunan görüntüler başlangıçta belirlenen target_size'a göre yeniden ölçeklendirilir.
    img = cv2.resize(img, target_size)

    # Ölçeklendirilen görüntüler x listesine eklenir.
    x.append(img)

    # Her bir path '/' ifadesi ile ayrıldığında dönen listenin sondan ikinci elemanı labelı temsil etmektedir.
    label = image_path.split(os.path.sep)[-2]

    # Yakalanan labelların sayısal değer karşılıklarının olduğu waste_labels sözlüğü içerisinden gönderilen key
    # değerine karşılık gelen value değeri alınarak label oluşturulur.
    labels.append(waste_labels[label])

    # Veri seti random bir şekilde karıştırılır.
    x, labels = shuffle(x, labels, random_state = 42)

    # Boyut ve sınıf bilgisi raporlanmaktadır.
  print(f"X boyutu: {np.array(x).shape}")
  print(f"Label sınıf sayısı: {len(np.unique(labels))} Gözlem sayısı: {len(labels)}")

  return x, labels
```
```python
x, labels = load_datasets(dir_path)
```
```python
# Görüntü boyutlarının tutulması
input_shape = (np.array(x[0]).shape[1],np.array(x[0]).shape[1], 3)
print(input_shape)
```
```python
def visualize_img(image_batch, label_batch):
  """
  Veri seti içerisinden görüntü görselleştirilir.

  Parametreler:

  image_batch: Görüntülere ait martris bilgilerini tutar.

  label_batch: Görüntünün ait olduğu sınıf bilgisini tutan liste.

  """
  plt.figure(figsize=(10, 10))
  for n in range(10):
    ax = plt.subplot(5,5,n+1)
    plt.imshow(image_batch[n])
    plt.title(np.array(list(waste_labels.keys()))[to_categorical(labels, num_classes=6)[n]==1][0].title())
    plt.axis('off')
```



# 3 Veriyi Hazırlamak (Data Preparation)

```python
# Train veri seti için bir generator tanımlıyoruz.
train = ImageDataGenerator(horizontal_flip=True,
                           vertical_flip=True,
                           validation_split=0.1,
                           rescale=1./255,
                           shear_range = 0.1,
                           zoom_range = 0.1,
                           width_shift_range= 0.1,
                           height_shift_range = 0.1)

# Test veri seti için bir generator tanımlıyoruz.
test = ImageDataGenerator(rescale=1/255,
                          validation_split=0.1)
```
```python
train_generator=train.flow_from_directory(directory=dir_path,
                                          target_size=(target_size),
                                          class_mode='categorical',
                                          subset='training')

test_generator=test.flow_from_directory(directory=dir_path,
                                        target_size=(target_size),
                                        batch_size=251,
                                        class_mode='categorical',
                                        subset='validation')
```



# 4) Modelleme (Modeling)

## 4.1) Sıfırdan CNN Modeli Kurma

- Sequential
- Evrişim Katmanı(Convolotion Layer,Conv2D)
- Havuzlama Katmanı (Pooling Layer)
- Aktivasyon Fonksiyonu Katmanı (Activation Layer)
- Flattening Katmanı
- Dense Katmanı
- Dropout Katmanı

```python
model=Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(input_shape), activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(input_shape), activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=(2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(input_shape), activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=(2,2)))

model.add(Flatten())

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=32, activation='relu'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=6, activation='softmax'))
```


## 4.2) Model Özeti

```python
model.summary()
```


## 4.3) Optimizasyon ve Değerlendirme Metriklerinin Ayarlanması

```python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "acc"])
```
```python
callbacks = [EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode="min"),
             
             ModelCheckpoint(filepath='mymodel.h5', monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=1)]
```


## 4.4) Modelin Eğitilmesi

```python
history = model.fit_generator(generator=train_generator,
                              epochs=15,
                              validation_data=test_generator,
                              callbacks=callbacks,
                              workers=4,
                              steps_per_epoch=2276//32,
                              validation_steps=251//32)
```


## 4.5) Accuracy ve Loss Grafikleri

```python
# Accuracy Grafiği

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label ='Training Accuracy')
plt.plot(history.history['val_acc'], label ='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy', fontsize=16)


# Loss Grafiği

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label ='Training Loss')
plt.plot(history.history['val_loss'], label ='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss', fontsize=16)
plt.show()
```



# 5) Değerlendirme (Evaluation)

```python
loss, precision, recall, acc = model.evaluate(test_generator, batch_size=32)
print("\nTest Accuracy: %.1f%%" % (100.0 * acc))
print("\nTest Loss: %.1f%%" % (100.0 * loss))
print("\nTest Precision: %.1f%%" % (100.0 * precision))
print("\nTest Recall: %.1f%%" % (100.0 * recall))
```
```python
# Classification Report
x_test, y_test = test_generator.next()

y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred, axis=1)

y_test = np.argmax(y_test, axis=1)

y_pred
```
```python
target_names = list(waste_labels.keys())
```
```python
print(classification_report(y_test, y_pred, target_names=target_names))
```
```python
# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  plt.figure(figsize=(8,6))
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True Label', fontweight="bold")
  plt.xlabel('Predict Label', fontweight="bold")
```
```python
plot_confusion_matrix(cm, waste_labels.keys(),
                      title='Confusion Matrix',
                      cmap=plt.cm.OrRd)
```



# Modelin Kullanılması / Test Edilmesi

```python
waste_labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
```
```python
def model_testing(path):
  """
  Görsellerin bulunduğu dizindeki görüntüyü okuyup model aracılığı ile hangi sınıfa ait olduğuna dair tahmin işlemi gerçekleştirilir.
  
  Parametreler:

  path: Görsellerin bulunduğu dizini ifade eder.

  Return:

  img: Görüntü

  p: Tahmin olasılıkları

  predicted_class: Tahmin sınıfı

  """

  img = image.load_img(path, target_size=(target_size))
  img = image.img_to_array(img, dtype=np.uint8)
  img=np.array(img)/255.0
  p=model.predict(img.reshape(1, 224, 224, 3))
  predicted_class = np.argmax(p[0])

  return img, p, predicted_class
```
```python
img1, p1, predicted_class1 = model_testing('/content/drive/MyDrive/Garbage_classification/metal/metal10.jpg')
img2, p2, predicted_class2 = model_testing('/content/drive/MyDrive/Garbage_classification/glass/glass105.jpg')
img3, p3, predicted_class3 = model_testing('/content/drive/MyDrive/Garbage_classification/cardboard/cardboard103.jpg')
img4, p4, predicted_class4 = model_testing('/content/drive/MyDrive/Garbage_classification/paper/paper106.jpg')

plt.figure(figsize=(20,60))

plt.subplot(141)
plt.axis('off')
plt.imshow(img1.squeeze())
plt.title("Maximum Probability: " + str(np.max(p1[0], axis = 0)) + "\n" + "Predicted class:" + str(waste_labels[predicted_class1]))
plt.imshow(img1);

plt.subplot(142)
plt.axis('off')
plt.imshow(img2.squeeze())
plt.title("Maximum Probability: " + str(np.max(p2[0], axis = 0)) + "\n" + "Predicted class:" + str(waste_labels[predicted_class2]))
plt.imshow(img2);

plt.subplot(143)
plt.axis('off')
plt.imshow(img3.squeeze())
plt.title("Maximum Probability: " + str(np.max(p3[0], axis = 0)) + "\n" + "Predicted class:" + str(waste_labels[predicted_class3]))
plt.imshow(img3);

plt.subplot(144)
plt.axis('off')
plt.imshow(img4.squeeze())
plt.title("Maximum Probability: " + str(np.max(p4[0], axis = 0)) + "\n" + "Predicted class:" + str(waste_labels[predicted_class4]))
plt.imshow(img4);
```

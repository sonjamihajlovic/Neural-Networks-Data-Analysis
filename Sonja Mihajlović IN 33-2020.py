# Import essential libraries
import requests
import cv2
import numpy as np
import imutils
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import load_model

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Spremi dataset na disk
np.savez('mnist.npz', train_images=train_images, train_labels=train_labels, test_images=test_images, test_labels=test_labels)

# %%
# Učitaj dataset
with np.load('mnist.npz') as data:
    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']


# Prikaz prvih 25 slika iz skupa za treniranje
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.ravel()

for i in np.arange(0, 25):
    axes[i].imshow(train_images[i], cmap='gray')
    axes[i].set_title(train_labels[i])
    axes[i].axis('off')



#%%

# Pretvaranje slika u 1D nizove i normalizacija podataka
train_images = train_images.reshape((train_images.shape[0], 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((test_images.shape[0], 28 * 28))
test_images = test_images.astype('float32') / 255



# Podela skupa za treniranje na skup za treniranje i skup za validaciju
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Kreiranje modela
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# #%%

# Kompajliranje modela
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treniranje modela
history = model.fit(train_images, train_labels, epochs=3, validation_data=(val_images, val_labels))

# Evaluacija modela
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# # Prikaz grafika kretanja funkcije gubitka i tačnosti
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='train_loss')
# plt.plot(history.history['val_loss'], label='val_loss')
# plt.title('Loss')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='train_acc')
# plt.plot(history.history['val_accuracy'], label='val_acc')
# plt.title('Accuracy')
# plt.legend()

# plt.show()

model.save('prethodno_obuceni_model.h5')
model = tf.keras.models.load_model('c:/FAX/3godina/semestar2/MITNOP/prethodno_obuceni_model.h5')



url = "http://172.20.10.6:8080/shot.jpg"


while True: 
	img_resp = requests.get(url)
	img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
	img = cv2.imdecode(img_arr, -1)
	img = imutils.resize(img, width=1000, height=1800)
	cv2.imshow("Android_cam", img)
 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	preprocessed_image = cv2.resize(gray, (28, 28))
	preprocessed_image = preprocessed_image.astype('float32') / 255
	preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    
	preprocessed_image = preprocessed_image.reshape(1, 28 * 28)
	predictions = model.predict(preprocessed_image)
	predicted_label = np.argmax(predictions[0])
	# Prikazivanje predikcije broja
	print('Predikcija broja:', predicted_label)

	cv2.imshow("Android_cam", img)
	
	if cv2.waitKey(1) == ord('q'):
		break

cv2.destroyAllWindows()

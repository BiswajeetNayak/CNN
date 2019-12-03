from keras.models import load_model
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

source = r'D:\Deep Learning\DL Datasets\cell_images'
destination = r'D:\Deep Learning\DL Datasets\cell_images\data'
DATADIR= r'D:\Deep Learning\DL Datasets\cell_images'

model = load_model(r'D:\Deep Learning\models\cell_classification_model_trained.h5')

test_file_names = [DATADIR+r"\data\test\Parasitized\\"+fname for fname in os.listdir(DATADIR+r"\data\test\Parasitized")] + [DATADIR+r"\data\test\Uninfected\\"+fname for fname in os.listdir(DATADIR+r"\data\test\Uninfected")]
X_test = [imageio.imread(image) for image in test_file_names]
X_test = np.array(X_test)
y_test = ["Parasitized"]*3000 + ["Uninfected"]*3000

print('Enter a number between 0 and 9999:')
index=int(input())
plt.imshow(X_test[index].reshape(128,128,3))
pred1 = model.predict(X_test.reshape(X_test.shape[0],128,128,3))
print(pred1[index])
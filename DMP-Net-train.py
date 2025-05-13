import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Sequential
from keras.layers import  Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Activation, Concatenate
import scipy.io as sio
from PIL import Image


def load_dataset():
    train_label = None
    train_input = None

    for scene_number in range(1, 34):
        if scene_number!= 27:
            dataset_dir1 = f"G:\\F盘的东西9.13\\脑瘤高光谱数据\\12.26实验数据\\rgb\\{scene_number}.jpg"
            dataset_dir2 = f"G:\\F盘的东西9.13\\脑瘤高光谱数据\\2.20\\duochidu\\img{scene_number}.mat"
            dataset_dir3 = f"G:\\F盘的东西9.13\\脑瘤高光谱数据\\12.26实验数据\\可见光mat\\scene{scene_number}.mat"

            img1 = np.array(Image.open(dataset_dir1))
            h, w, nC1 = img1.shape
            img1 = img1.reshape((h * w, nC1))

            data2 = sio.loadmat(dataset_dir2)
            img2 = data2['img']/1
            # print(img2)
            h, w, nC2 = img2.shape
            img2 = img2.reshape((h * w, nC2))

            combined_img = np.concatenate((img1, img2), axis=1)

            print(combined_img.shape)

            label_data = sio.loadmat(dataset_dir3)
            img3 = label_data['scene1'] / 255
            # print(img3)
            h, w, nC3 = img3.shape
            img3 = img3.reshape((h * w, nC3))

            if train_input is None:
                train_input = combined_img
                train_label = img3
            else:
                train_input = np.vstack((train_input, combined_img))
                train_label = np.vstack((train_label, img3))

    # train_label = train_label[:, :14]
    test_label = train_label.copy()

    test_input = train_input.copy()

    dataset = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }

    return dataset

def load_dataset1(I, J):

    train_input = None

    for scene_number in range(I, J):

        dataset_dir11 = f"G:\\F盘的东西9.13\\脑瘤高光谱数据\\12.26实验数据\\rgb\\{scene_number}.jpg"
        dataset_dir12 = f"G:\\F盘的东西9.13\\脑瘤高光谱数据\\2.20\\duochidu\\img{scene_number}.mat"

        img1 = np.array(Image.open(dataset_dir11))
        h, w, nC1 = img1.shape
        img1 = img1.reshape((h * w, nC1))

        data2 = sio.loadmat(dataset_dir12)
        img2 = data2['img'] / 255
        print(img2)
        h, w, nC2 = img2.shape
        img2 = img2.reshape((h * w, nC2))

        combined_img1 = np.concatenate((img1, img2), axis=1)

        if train_input is None:
            train_input = combined_img1
        else:
            train_input = np.vstack((train_input, combined_img1))

    dataset = {
        'train_input': train_input,
    }

    return dataset


dataset = load_dataset()
df = dataset['train_input']
df2 = dataset['train_label']

# # 提取特征和标签
X = df[:, 0:6]
Y = df2[:, 0:69]

# print(Y.shape)
X_train=X
rows, columns = X_train.shape
Y_train=Y
X_test=X
Y_test=Y
X_train = X_train.reshape(rows,1,6)
X_test = X_test.reshape(rows,1,6)

# 创建模型
model = Sequential()

model.add(Conv1D(filters=16,kernel_size=3,batch_input_shape=(None,1,6),strides=1,padding='same',activation='linear'))
model.add(Conv1D(16, 3, strides=1, padding='same',activation='linear'))
model.add(MaxPooling1D(3, 1, 'same'))

merge_layer1 = model.layers[0].output
merge_layer2 = model.layers[1].output

merge_layer = Concatenate()([merge_layer1, merge_layer2])

model.add(Conv1D(64,3, strides=1, padding='same',activation='linear'))
model.add(Conv1D(64, 3, strides=1, padding='same',activation='linear'))
model.add(MaxPooling1D(3, 1, 'same'))

merge_layer3 = model.layers[3].output
merge_layer = Concatenate()([merge_layer, merge_layer3])

model.add(Conv1D(256,3, strides=1, padding='same',activation='linear'))
model.add(Conv1D(256, 3,strides=1, padding='same',activation='linear'))
model.add(MaxPooling1D(3, 1, 'same'))

merge_layer4 = model.layers[6].output
merge_layer = Concatenate()([merge_layer, merge_layer4])

model.add(Conv1D(512,3, strides=1, padding='same',activation='linear'))
model.add(Conv1D(512, 3,strides=1, padding='same',activation='linear'))
model.add(MaxPooling1D(3, 1, 'same'))

merge_layer5 = model.layers[9].output
merge_layer = Concatenate()([merge_layer, merge_layer5])

model.add(Conv1D(128, 3, strides=1, padding='same',activation='linear'))
model.add(Conv1D(128, 3, strides=1, padding='same',activation='linear'))
model.add(MaxPooling1D(3, 1, 'same'))

model.add(Flatten())
model.add(Dense(69, activation='linear'))

def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[coeff_determination])

model.fit(X_train,Y_train, validation_data=(X_test, Y_test),epochs=10, batch_size=128)


model_json = model.to_json()
with open(r"C:\Users\Desktop\model.json",'w')as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

#---------------------------------test-------------------------------#

test = load_dataset1(27, 28 )
df3 = test['train_input']
X_test1 = df3[:, 0:6]
rows, columns = X_test1.shape
print(rows)
X_test=X_test1
X_test = X_test.reshape(rows,1,6)

predicted = model.predict(X_test)
print('predicted:', predicted.shape)

new_ = pd.DataFrame(predicted)
new_.to_csv(r".\result\cnn27.csv", header=False, index=False)

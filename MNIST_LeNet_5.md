

```python
import keras
from keras.datasets import mnist
from keras.models import Sequential
# 各个层有不同的作用 依次为 全连接层  
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from keras import backend as K
```

    Using TensorFlow backend.
    


```python
num_classes = 10
img_rows,img_cols = 28, 28
```


```python
(trainX, trainY),(testX,testY) = mnist.load_data()
```

    Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
    11493376/11490434 [==============================] - 1s 0us/step
    


```python
if K.image_data_format() == 'channals_first':
  trainX = trainX.reshape(trainX.shape[0],1,img_rows,img_cols)
  testX = testX.reshape(testX.shape[0],1,img_rows,img_cols)
  input_shape = (1,img_rows, img_cols)
else:
  trainX = trainX.reshape(trainX.shape[0],img_rows,img_cols,1)
  testX = testX.reshape(testX.shape[0],img_rows,img_cols,1)
  input_shape = (img_rows, img_cols,1)
 
```


```python
# 将图像像素转换为0-1之间实数
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX = trainX / 255.0
testX = testX / 255.0
```


```python
# 将标准答案 转为 one-hot向量。例如数字1 就是  [0,1,0,0,0,0,0,0,0,0] 
trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)
```


```python
model = Sequential()
# 卷积层 需要定义输入shape=[28,28,1] 默认不padding  默认步长为1
model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=input_shape))
# 卷积后 尺寸为 （24,24,32）(28-5+1)/1 = 24 加快计算 防止过拟合
# 最大池化层 
model.add(MaxPooling2D(pool_size=(2,2)))
# 池化后 （12,12,32）
# 继续卷积 
model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
# 卷积后 (8,8,64) 
# 继续池化
model.add(MaxPooling2D(pool_size=(2,2)))
# 池化后 尺寸为  (4,4,64)
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Flatten())
# Flatten之后 为 4*4*16 256
# 全连接层 500个结点 激活函数为 relu
model.add(Dense(500,activation='relu'))
# 全连接层 也是 输出层  10个结点  
model.add(Dense(num_classes,activation='softmax'))
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    


```python
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer=keras.optimizers.SGD(),
             metrics=['accuracy'])
```


```python
model.fit(trainX,trainY,batch_size=128,
         epochs=20,
         validation_data=(testX,testY))
```

    Train on 60000 samples, validate on 10000 samples
    Epoch 1/20
    60000/60000 [==============================] - 2s 41us/step - loss: 0.0638 - acc: 0.9809 - val_loss: 0.0554 - val_acc: 0.9834
    Epoch 2/20
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0598 - acc: 0.9821 - val_loss: 0.0546 - val_acc: 0.9832
    Epoch 3/20
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0564 - acc: 0.9830 - val_loss: 0.0503 - val_acc: 0.9858
    Epoch 4/20
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0535 - acc: 0.9839 - val_loss: 0.0514 - val_acc: 0.9838
    Epoch 5/20
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0504 - acc: 0.9853 - val_loss: 0.0443 - val_acc: 0.9864
    Epoch 6/20
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0485 - acc: 0.9854 - val_loss: 0.0434 - val_acc: 0.9866
    Epoch 7/20
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0468 - acc: 0.9859 - val_loss: 0.0486 - val_acc: 0.9856
    Epoch 8/20
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0443 - acc: 0.9869 - val_loss: 0.0440 - val_acc: 0.9852
    Epoch 9/20
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0428 - acc: 0.9871 - val_loss: 0.0474 - val_acc: 0.9828
    Epoch 10/20
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0405 - acc: 0.9878 - val_loss: 0.0411 - val_acc: 0.9868
    Epoch 11/20
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0398 - acc: 0.9883 - val_loss: 0.0396 - val_acc: 0.9875
    Epoch 12/20
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0379 - acc: 0.9892 - val_loss: 0.0429 - val_acc: 0.9876
    Epoch 13/20
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0366 - acc: 0.9890 - val_loss: 0.0387 - val_acc: 0.9874
    Epoch 14/20
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0355 - acc: 0.9895 - val_loss: 0.0363 - val_acc: 0.9883
    Epoch 15/20
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0343 - acc: 0.9897 - val_loss: 0.0356 - val_acc: 0.9884
    Epoch 16/20
    60000/60000 [==============================] - 2s 39us/step - loss: 0.0328 - acc: 0.9904 - val_loss: 0.0345 - val_acc: 0.9894
    Epoch 17/20
    60000/60000 [==============================] - 2s 40us/step - loss: 0.0323 - acc: 0.9907 - val_loss: 0.0359 - val_acc: 0.9884
    Epoch 18/20
    60000/60000 [==============================] - 3s 44us/step - loss: 0.0310 - acc: 0.9908 - val_loss: 0.0337 - val_acc: 0.9885
    Epoch 19/20
    60000/60000 [==============================] - 3s 44us/step - loss: 0.0300 - acc: 0.9910 - val_loss: 0.0397 - val_acc: 0.9865
    Epoch 20/20
    60000/60000 [==============================] - 3s 44us/step - loss: 0.0291 - acc: 0.9915 - val_loss: 0.0338 - val_acc: 0.9886
    




    <keras.callbacks.History at 0x7fe384b66160>




```python
score = model.evaluate(testX,testY)
print('Test loss:',score[0])
print('test accuracy',score[1])
```

    10000/10000 [==============================] - 0s 49us/step
    Test loss: 0.03384795787587063
    test accuracy 0.9886
    

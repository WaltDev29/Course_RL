# MNIST 데이터로 신경망의 일반화 능력 측정
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as ds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

(x_train,y_train),(x_test,y_test)=ds.mnist.load_data() # MNIST 데이터 준비
xx_train=x_train.reshape(60000,784)
xx_test=x_test.reshape(10000,784)
xx_train=xx_train.astype(np.float32)/255.0
xx_test=xx_test.astype(np.float32)/255.0
yy_train=tf.keras.utils.to_categorical(y_train,10)
yy_test=tf.keras.utils.to_categorical(y_test,10)

mlp=Sequential() # MLP 구현
mlp.add(Dense(units=512,activation='relu',input_shape=(784,)))
mlp.add(Dense(units=10,activation='softmax'))

mlp.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
hist=mlp.fit(xx_train,yy_train,batch_size=128,epochs=50,validation_data=(xx_test,yy_test),verbose=2)

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy graph')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.grid()
plt.show()

res=mlp.evaluate(xx_test,yy_test,verbose=0) # 테스트 집합에 대한 정확률 평가
print('테스트 집합에 대한 정확률=',res[1]*100)

y=mlp.predict(xx_test[0].reshape(1,784),verbose=0) # 첫 번째 테스트 샘플에 대한 출력 확인
print('y=',y)

plt.figure(figsize=(18,8)) # 테스트 집합에서 앞쪽 40개 샘플을 예측하고 디스플레이
for i in range(40):
    plt.subplot(4,10,i+1)
    plt.imshow(x_test[i],cmap='gray')
    plt.xticks([]);plt.yticks([])
    y=mlp.predict(xx_test[i].reshape(1,784),verbose=0)
    plt.title(str(y_test[i])+'->'+str(np.argmax(y)),fontdict={'fontsize':24})
    plt.show()

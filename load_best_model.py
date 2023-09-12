#Hàm load model tốt nhất dựa trên tập test của fashion_mnist
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from sklearn.metrics import accuracy_score
def evaluate_model(model, x_test,y_test):
    y_pred=np.argmax(model.predict(x_test), axis=1)
    return accuracy_score(y_test,y_pred)
def evaluate_folder_model(model_dir):
    max_accuracy=0
    max_model_path=''
    fashion_set=fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = fashion_set
    x_test=np.array(x_test,dtype = 'float32')
    x_test=x_test/255.0
    x_test=x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_channel_1=x_test
    x_channel_3=x_test=x_test.repeat(3, axis=-1)
    for file in os.listdir(model_dir):
        if '.'in file:
            print(f'Found file: {file}')
            if 'transfer' in file:
                x_test=x_channel_3
            else:
                x_test=x_channel_1
            model_path=os.path.join(model_dir,file)
            model=keras.models.load_model(model_path)
            accuracy=evaluate_model(model,x_test,y_test)
            print('accuracy: ',end='')
            print(accuracy)
            if accuracy>max_accuracy:
                max_accuracy=accuracy
                max_model_path=model_path
        else:
            print(f'Found folder (not load): {file}')
    return max_model_path,max_accuracy
if __name__=='__main__':
    model_dir=r'D:\PythonWorkspace\PythonMain\MNIST_DOSOMETHING\FashionMNIST\model_training'
    print(evaluate_folder_model(model_dir))

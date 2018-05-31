from zipfile import ZipFile
import numpy as np

'''load your data here'''

class DataLoader(object):
    def __init__(self):
        DIR = '../data/'
        pass
    
    # Returns images and labels corresponding for training and testing. Default mode is train. 
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode = 'train'):
        label_filename = mode + '_labels'
        image_filename = mode + '_images'
        label_zip = '../data/' + label_filename + '.zip'
        image_zip = '../data/' + image_filename + '.zip'
        with ZipFile(label_zip, 'r') as lblzip:
            labels = np.frombuffer(lblzip.read(label_filename), dtype=np.uint8, offset=8)
        with ZipFile(image_zip, 'r') as imgzip:
            images = np.frombuffer(imgzip.read(image_filename), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        return images, labels

    def create_batches(self, X, Y, batch_size = 100):
        randomArray = np.arange(X.shape[0])
        np.random.shuffle(randomArray)
        X, Y = X[randomArray], Y[randomArray]
        X_batch = []
        y_batch = []
        for index_value in range(0, X.shape[0], batch_size):
            X_batch.append(X[index_value:index_value+batch_size])
            y_batch.append(Y[index_value:index_value+batch_size])
        return X_batch, y_batch


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from keras.datasets import cifar10

import time


from keras.preprocessing.image import ImageDataGenerator

import pickle
import numpy as np


print('Data augmentation with keras.image')
gen = ImageDataGenerator(
    #samplewise_std_normalization=True, # global contrast normalization
    #samplewise_center=True,
    #featurewise_center=False,
    #featurewise_std_normalization=False,
    zca_whitening=True,
    zca_epsilon=1e-6,
    #shear_range=0.2,
    #channel_shift_range=0.2,
    #fill_mode='nearest',
    #rotation_range=2,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    #horizontal_flip=True,
    #vertical_flip=False
)


(images_train, y_train), (images_test, labels_test) = cifar10.load_data()

epochs = 1
batch_size = 100
num_batches = (len(images_train)*epochs // batch_size)
batches = 0
t = time.time()
t0 = time.time()

t_print = 10

os.makedirs(parentdir + '/cifar10_data' + '/augmented_data', exist_ok=True)
save_dir = parentdir + '/cifar10_data' + '/augmented_data'
images_aug_pickle = save_dir +'/images.pickle'
y_aug = save_dir + '/y.pickle'


# include original data
'''
with open(images_aug_pickle, 'wb') as handle:
    pickle.dump(images_train, handle)

with open(y_aug, 'wb') as handle:
    pickle.dump(y_train, handle)
'''

gen.fit(images_train)
'''
X_train_aug=np.empty((0, *images_train.shape[1:]))
y_train_aug=np.empty((0, *y_train.shape[1:]))

'''
handle1 = open(images_aug_pickle, 'a+b')
handle2 = open(y_aug, 'a+b')

for X_batch, y_batch in gen.flow(np.repeat(images_train, epochs, axis=0), np.repeat(y_train, epochs,axis= 0), batch_size=batch_size):

    #X_train_aug = np.append(X_train_aug, X_batch, axis=0)
    #y_train_aug = np.append(y_train_aug, y_batch, axis=0)
    pickle.dump(X_batch,  open(images_aug_pickle, 'ab'))
    pickle.dump(y_batch, open(y_aug, 'ab') )

    # report time
    eta = int(((time.time() - t0) / (batches + 1)) * (num_batches - batches - 1))
    if time.time() - t > t_print:
        print('processing image batch %d/%d, eta: %d:%02d:%02d' % (
            batches, num_batches, eta / 3600, (eta % 3600) / 60, eta % 60))
        t = time.time()

    batches += 1
    if batches >= num_batches:
        break

'''
with open(images_aug_pickle, 'wb') as handle:
    pickle.dump(X_train_aug, handle)

with open(y_aug, 'wb') as handle:
    pickle.dump(y_train_aug, handle)
'''



def main():

    # load augmented dataset
    images_aug= np.empty((0, *images_train.shape[1:]))
    y_augm = np.empty((0, *y_train.shape[1:]))

    with open(images_aug_pickle, 'rb') as handle:
        while 1:
            try:
                images_aug = np.append(images_aug,pickle.load(handle), axis=0)
            except EOFError:
                break



    with open(y_aug, 'rb') as handle:
        while 1:
            try:
                y_augm = np.append(y_augm,pickle.load(handle), axis=0)
            except EOFError:
                break


    from images_utils import plot_images


    plot_images(images=images_aug[0:9],
                cls_true=y_augm[0:9])


    import hpelm
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_train = onehot_encoder.fit_transform(y_augm)
    X_train = images_aug.reshape((-1,32*32*3))

    print('\nELM simple')
    elm = hpelm.ELM(X_train.shape[1], 10, classification="c", accelerator="GPU", precision='single')
    elm.add_neurons(8196, 'sigm')
    print(str(elm))
    # Training model
    t = time.time()
    elm.train(X_train, Y_train, 'c')
    elapsed_time_train = time.time() - t
    y_train_predicted = elm.predict(X_train)
    print("Training time: %f" % elapsed_time_train)
    print('Training Accuracy: ', (1 - elm.error(Y_train, y_train_predicted)))

    print(2)


if __name__ == '__main__':
    main()

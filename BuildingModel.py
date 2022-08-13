# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 10:36:54 2022

@author: 4hmet
"""
# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss

# Computing Precision 
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

IMG_SIZE=128 #for 128x128
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate,  BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.metrics import MeanIoU
from keras.layers import UpSampling2D
from tensorflow.keras import regularizers
"""
def build_unet(inputs, ker_init,dropout):
    con= Conv2D(16, 3, activation='relu',padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init) (inputs)
    con=Dropout(dropout) (con)
    con=Conv2D(16,3,activation='relu',padding = 'same', kernel_regularizer=regularizers.l2(0.0001),kernel_initializer = ker_init) (con)

    fpool= MaxPooling2D(pool_size=(2, 2))(con)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(fpool)
    conv1= Dropout(dropout) (conv1)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(conv1)
    
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001),kernel_initializer = ker_init)(pool)
    conv2= Dropout(dropout) (conv2)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001),kernel_initializer = ker_init)(conv2)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(pool1)
    conv3= Dropout(dropout) (conv3)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(conv3)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001),  kernel_initializer = ker_init)(pool2)
    conv4= Dropout(dropout) (conv4)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(conv4)
    
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(pool4)
    conv5= Dropout(dropout) (conv5)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(conv5)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv5))
    merge7 = concatenate([conv4,up7],axis=3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(merge7)
    conv7=Dropout(dropout) (conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv3,up8],axis=3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(merge8)
    conv8= Dropout(dropout) (conv8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001),kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv2,up9],axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001),kernel_initializer = ker_init)(merge9)
    conv9= Dropout(dropout) (conv9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001),kernel_initializer = ker_init)(conv9)
    
    up = Conv2D(32, 2, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv9))
    merge = concatenate([conv1,up],axis= 3 )
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(merge)
    conv= Dropout(dropout) (conv)
    conv = Conv2D(32, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(conv)
    
    ups=Conv2D(16,2,activation = 'relu', padding = 'same', kernel_regularizer=regularizers.l2(0.0001),kernel_initializer = ker_init)(UpSampling2D(size = (2,2))(conv))
    merges=concatenate([con,ups],axis=3 )
    convs= Conv2D(16,3,activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001), kernel_initializer = ker_init)(merges)
    convs= Dropout(dropout)(convs)
    convs= Conv2D(16, 3, activation = 'relu', padding = 'same',kernel_regularizer=regularizers.l2(0.0001),  kernel_initializer = ker_init)(convs)
    
    conv10 = Conv2D(4, (1,1), activation = 'softmax')(convs)
    
    return Model(inputs = inputs, outputs = conv10)

input_layer = Input((IMG_SIZE, IMG_SIZE, 2))

model = build_unet(input_layer, 'he_normal', 0.2)
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision])
"""
#Load data
#Loading all data into memory is not a good idea since the data are too big to fit in. So we will create dataGenerators - load data on the fly as explained here
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
def pathListIntoIds(dirList):
    x = []
    for i in range(0,len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x

train_and_test_ids = pathListIntoIds(train_and_val_directories); 

    
train_test_ids, val_ids = train_test_split(train_and_test_ids,test_size=0.2) 
train_ids, test_ids = train_test_split(train_test_ids,test_size=0.15)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, dim=(IMG_SIZE,IMG_SIZE), batch_size = 1, n_channels = 2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size*VOLUME_SLICES, *self.dim, 4))

        
        # Generate data
        for c, i in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            data_path = os.path.join(case_path, f'{i}_flair.nii');
            flair = nib.load(data_path).get_fdata()    
            
            data_path = os.path.join(case_path, f'{i}_t1ce.nii');
            t1ce = nib.load(data_path).get_fdata()
            
            data_path = os.path.join(case_path, f'{i}_seg.nii');
            seg = nib.load(data_path).get_fdata()
        
            for j in range(VOLUME_SLICES):
                 X[j +VOLUME_SLICES*c,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
                 X[VOLUME_SLICES*c,:,:,1] = cv2.resize(t1ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));

                 
                 y[j +VOLUME_SLICES*c] = seg[:,:,j+VOLUME_START_AT];
                    
        # Generate masks
        y[y==4] = 3;
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE));
        return X/np.max(X), Y #min max scaling
        
training_generator = DataGenerator(train_ids)
valid_generator = DataGenerator(val_ids)
test_generator = DataGenerator(test_ids)

# show number of data for each dir 
def showDataLayout():
    plt.bar(["Train","Valid","Test"],
    [len(train_ids), len(val_ids), len(test_ids)], align='center',color=[ 'green','red', 'blue'])
    plt.legend()

    plt.ylabel('Number of images')
    plt.title('Data distribution')

    plt.show()
    
showDataLayout()


#Add callback for training process


csv_logger = CSVLogger('training.log', separator=',', append=False)


callbacks = [
     #keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                               #patience=2, verbose=1, mode='auto'),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1),
#keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
 #                            verbose=1, save_best_only=True, save_weights_only = True),
        csv_logger
    ]
"""
history =  model.fit(training_generator,
                     epochs=35,
                     steps_per_epoch=len(train_ids),
                     callbacks= callbacks,
                     validation_data = valid_generator
                     )  
model.save("model_x9_9.h5")

#K. clear_session()
"""
model = keras.models.load_model('C:/Users/4hmet/BraTS2020/model_x2_2.h5', 
                                   custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": dice_coef,
                                                   "precision": precision,
                                                  }, compile=False)
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision])
history=model.fit(training_generator,
                     epochs=35,
                     steps_per_epoch=len(train_ids),
                     callbacks= callbacks,
                     validation_data = valid_generator
                     )
history1 = pd.read_csv('C:/Users/4hmet/BraTS2020/training.log', sep=',', engine='python')

hist=history1

acc=hist['accuracy']
val_acc=hist['val_accuracy']

epoch=range(len(acc))

loss=hist['loss']
val_loss=hist['val_loss']

train_dice=hist['dice_coef']
val_dice=hist['val_dice_coef']

f,ax=plt.subplots(1,4,figsize=(16,8))

ax[0].plot(epoch,acc,'b',label='Training Accuracy')
ax[0].plot(epoch,val_acc,'r',label='Validation Accuracy')
ax[0].legend()

ax[1].plot(epoch,loss,'b',label='Training Loss')
ax[1].plot(epoch,val_loss,'r',label='Validation Loss')
ax[1].legend()

ax[2].plot(epoch,train_dice,'b',label='Training dice coef')
ax[2].plot(epoch,val_dice,'r',label='Validation dice coef')
ax[2].legend()

ax[3].plot(epoch,hist['mean_io_u_5'],'b',label='Training mean IOU')
ax[3].plot(epoch,hist['val_mean_io_u_5'],'r',label='Validation mean IOU')
ax[3].legend()
conda install graphviz
import keras.utils.vis_utils
from importlib import reload
reload(keras.utils.vis_utils)


from keras.utils.vis_utils import plot_model    
plot_model(model, to_file='BestModel_plot.png')

model.summary()
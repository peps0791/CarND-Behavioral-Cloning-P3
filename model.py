import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout
from keras.models import Sequential
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import cv2

DATA_DIRECTORY = 'data/'
TRAINING_SPLIT = 0.8
BATCH_SIZE = 32
LEARNING_RATE = 1.0e-4
EPOCHS = 10
MODEL_NAME = 'sample_data_new_model_filter'
MODEL_ROW_SIZE = 64
MODEL_COL_SIZE = 64
NO_OF_CHANNELS = 3
MODEL_INPUT_SHAPE = (MODEL_ROW_SIZE, MODEL_COL_SIZE, NO_OF_CHANNELS)
TARGET_SIZE = (MODEL_ROW_SIZE, MODEL_COL_SIZE)
STEERING_CORRECTION = 0.25
STEERING_THRESHOLD = 0.15
STEERING_KEEP_PROBABILITY_THRESHOLD = 1
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160


def convert_to_YUV(image):
    '''converts the image from RGB space to YUV space'''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def resize_image(image):
    '''resize the image to 64 by 64 size'''
    return cv2.resize(image, TARGET_SIZE)

def crop_image(image):
    '''
    :param image: The input image of dimensions 160x320x3
    :crops out the sky and bumper portion of the car form the image
    '''
    cropped_image = image[55:135, :, :]
    return cropped_image

def translate_image(image, steering_angle, range_x=100, range_y=10):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def preprocess_image(image):
    '''does the crop, resize and conversion to YUV space of the image'''
    image = crop_image(image)
    image = resize_image(image)
    image = convert_to_YUV(image)
    image = np.array(image)
    return image

def choose_image(row_data):
    '''Selects an image from amongst the center camera image, left camra image 
    and the rigt camera image '''
    toss = np.random.randint(3)
    img_path = ''
    steering = 0.0
    if toss == 0:
        #choose center image
        img_path = row_data.iloc[0]
        steering = row_data.iloc[3]
    elif toss==1:
        #choose left image
        img_path = row_data.iloc[1]
        steering = row_data.iloc[3] + STEERING_CORRECTION
    elif toss==2:
        #choose right image
        img_path = row_data.iloc[2]
        steering = row_data.iloc[3] - STEERING_CORRECTION
    return img_path, steering

def load_image(img_path):
    '''loads an image by reading from a file path'''
    img_path = img_path.strip()
    #print('image path-->', DATA_DIRECTORY + img_path)
    image = cv2.imread(DATA_DIRECTORY + img_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    return image

def bright_augment_image(img):
    '''Adds rando mbrightness to the image'''
    img1 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    random_bright = .25 + np.random.uniform()
    img1[:,:,2] = img1[:,:,2]*random_bright
    img1 = cv2.cvtColor(img1,cv2.COLOR_HSV2RGB)
    return img1

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def augment_image(row_data):
    '''auments the input image by translating, adding random shadow, augmenting
    brightness and preprocessing (crop , resize and conversion to YUV space)
    the image''' 
    img_path, steering = choose_image(row_data)
    image = load_image(img_path)
    #translate image
    image, steering = translate_image(image, steering)
    
    #augment brightness
    image = bright_augment_image(image)

    #flip image
    # This is done to reduce the bias for turning left that is present in the training data
    flip_prob = np.random.random()
    if flip_prob > 0.5:
        # flip the image and reverse the steering angle
        steering = -1*steering
        image = cv2.flip(image, 1)

    #random shadow
    image = random_shadow(image)

    #preprocess image
    image = preprocess_image(image)
   
    return image, steering

def get_data_generator(data_frame):
    '''generator to generate data for the model on the fly.'''
    
    batch_size = BATCH_SIZE
    print('data generator called')
    N = data_frame.shape[0]
    print('N -->', N)
    batches_per_epoch = N // batch_size
    print('batches per epoch-->', batches_per_epoch)

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # slice a `batch_size` sized chunk from the dataframe
        # and generate augmented data for each row in the chunk on the fly
        for index, row in data_frame.loc[start:end].iterrows():
            image, steering =  augment_image(row)
            X_batch[j]  = image
            y_batch[j] = steering
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch

def main():
    '''main function from where the code flow starts'''
    print('Udacity Project 3: Behavioral Cloning')
    org_data_frame = load_data()
    print('Data loaded')
    #get_data_stats(org_data_frame)

    train_data, valid_data, new_data_frame = filter_dataset(org_data_frame)
    print('Data filtered')
    
    #get_data_stats(new_data_frame)
    org_data_frame = None

    print('calling generators')
    training_generator = get_data_generator(train_data) 
    validation_data_generator = get_data_generator(valid_data)
   
    model = get_model()

    adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = 'adam', loss = 'mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
    save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

    model.fit_generator(training_generator, validation_data=validation_data_generator, nb_epoch = EPOCHS, 
    callbacks=[save_weights, early_stopping], samples_per_epoch = 20000, nb_val_samples = len(valid_data))
    #model.fit(np.array([[1]]), np.array([[2]]), batch_size=BATCH_SIZE, shuffle = True, nb_epoch = EPOCHS, callbacks=[save_weights, early_stopping])
    
    #save the model
    model.save(MODEL_NAME)
    print("Model Saved!")

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255., input_shape=MODEL_INPUT_SHAPE, name='normalizer'))

    '''model.add(Cropping2D(cropping=crop,
                             input_shape=(row, col, ch), name='cropping'))'''

    model.add(Conv2D(3, 1, 1, subsample=(2, 2), activation='elu', name='cv0'))

    model.add(Conv2D(16, 3, 3, activation='elu', name='cv1'))
    model.add(MaxPooling2D(name='maxPool_cv1'))
    model.add(Dropout(0.5, name='dropout_cv1'))

    model.add(Conv2D(32, 3, 3, activation='elu', name='cv2'))
    model.add(MaxPooling2D(name='maxPool_cv2'))
    model.add(Dropout(0.5, name='dropout_cv2'))

    model.add(Conv2D(64, 3, 3, activation='elu', name='cv3'))
    model.add(MaxPooling2D(name='maxPool_cv3'))
    model.add(Dropout(0.5, name='dropout_cv3'))

    model.add(Flatten())

    model.add(Dense(1000, activation='elu', name='fc1'))
    model.add(Dropout(0.5, name='dropout_fc1'))

    model.add(Dense(100, activation='elu', name='fc2'))
    model.add(Dropout(0.5, name='dropout_fc2'))

    model.add(Dense(10, activation='elu', name='fc3'))
    model.add(Dropout(0.5, name='dropout_fc3'))

    model.add(Dense(1, name='output'))
    return model

#even out the steering angle values in data
#filter out the 0 and lower speed values from data
def filter_dataset(data_frame):
    data_frame = filter_steering(data_frame)
    #filter_throttle()
    num_rows_training = int(data_frame.shape[0]*TRAINING_SPLIT)

    training_data = data_frame.loc[0:num_rows_training-1]
    validation_data = data_frame.loc[num_rows_training:]
    return training_data, validation_data, data_frame

def filter_steering(data_frame):
    '''evens out the steering angle distribution in the data set by removing 
    70% of the rows with steering angle close to 0 (<0.01).'''
    print('Steering filtering')
    print ('number of rows with steering less than 0.01: ', len(data_frame.loc[data_frame['steering'] <0.007]))
    data_frame = data_frame.drop(data_frame[data_frame['steering'] <0.01].sample(frac=0.70).index)
    print('new data frame length-->', len(data_frame))
    return data_frame

def filter_throttle():
    '''filters out the throttle values less than 0.25'''
    print('Throttle filtering')
    #print (len(org_data_frame.loc[org_data_frame['throttle'] >=0.5]))

def load_data():
    '''loads the data from the driving log .csv file into a dataframe'''
    data_frame = pd.read_csv(DATA_DIRECTORY +'driving_log.csv')

    # shuffle the data
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    # 80-20 training validation split
    '''num_rows_training = int(data_frame.shape[0]*TRAINING_SPLIT)

    training_data = data_frame.loc[0:num_rows_training-1]
    validation_data = data_frame.loc[num_rows_training:]'''
    return data_frame

if __name__ == '__main__':
    main()
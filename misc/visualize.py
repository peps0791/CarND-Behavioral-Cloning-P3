import cv2
import numpy as np
from model import preprocess_image
from model import translate_image
from model import bright_augment_image
from model import random_shadow
from model import load_data
from model import filter_dataset
from model import crop_image

def show_images(data_frame):

    for index, row in data_frame[:1].iterrows():
        img = row.iloc[0]

        print('img-->', img)
        image = cv2.imread('data/'+img)
        cv2.imshow("original image", image)
        #cv2.waitKey(0)
        cv2.imwrite('org_image.png', image)

        cropped_image = crop_image(image)
        cv2.imshow("cropped image", cropped_image)
        #cv2.waitKey(0)
        cv2.imwrite('cropped_image.png', cropped_image)

        
        flip_image = cv2.flip(image, 1)
        cv2.imshow("flipped image", flip_image)
        #cv2.waitKey(0)
        cv2.imwrite('flip_image.png', flip_image)

        org_image = image
        for i in range(5):
            image, steering = show_translated_images(org_image)
            cv2.imshow("translated image-" + str(i), image)
            #cv2.waitKey(0)
            cv2.imwrite('transated_image-' + str(i)+'.png', image)

        org_image = image
        for i in range(5):
            image = show_augmented_brightness_images(org_image)
            cv2.imshow("augmented brightness image-" + str(i), image)
            #cv2.waitKey(0)
            cv2.imwrite('bright_image-' + str(i)+'.png', image)

        org_image = image
        for i in range(5):
            image = random_shadow(org_image)
            cv2.imshow("random shadow image-" + str(i), image)
            #cv2.waitKey(0)
            cv2.imwrite('shadow_image-' + str(i)+'.png', image)


def show_translated_images(image):
    return translate_image(image, np.random.uniform(low=-1.0, high=1.0))

def show_augmented_brightness_images(image):
    return bright_augment_image(image)

def show_random_shadow_images(image):
    return random_shadow(image)

def proprocessed_image(image):
    return preprocess_image(image)

data_frame = load_data()
train, valid, data_frame = filter_dataset(data_frame)
show_images(data_frame)

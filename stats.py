import numpy as np
import matplotlib.pyplot as plt
import cv2
from model import load_data
from model import filter_dataset

def get_steering_angle_stats(data_frame):
    print('Steering angle distribution statistics')
    steering_val = data_frame.iloc[:,[3]].as_matrix()

    hist, bins = np.histogram(steering_val, bins=1000)
    center = (bins[:-1] + bins[1:]) / 2
    width = 20 * (bins[1] - bins[0])
    fig = plt.figure()
    fig.suptitle('Steering angle distribution', fontsize=20)
    plt.xlabel('Steering angles range', fontsize=18)
    plt.ylabel('Frequency', fontsize=16)
    plt.bar(center, hist, align='center', width = width)
    plt.show()
    

def get_speed_stats(data_frame):
    print('Speed distribution statistics')
    speed_val = data_frame.iloc[:,[6]].as_matrix()

    hist, bins = np.histogram(speed_val, bins=100)
    center = (bins[:-1] + bins[1:]) / 2
    width = 2 * (bins[1] - bins[0])
    fig = plt.figure()
    fig.suptitle('Speed Distribution', fontsize=20)
    plt.xlabel('Speed range', fontsize=18)
    plt.ylabel('Frequency', fontsize=16)
    plt.bar(center, hist, align='center', width = width)
    plt.show()

    '''
    #show images where speed is the lowest
    lowest_speed_dataframe = data_frame.nsmallest(5, 'speed')
    for index, row in lowest_speed_dataframe.iterrows():
        img = row.iloc[0]
        image = cv2.imread('data/'+img)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        #print(img)'''

def get_throttle_stats(data_frame):
    print('Throttle distribution statistics')
    t_val = data_frame.iloc[:,[4]].as_matrix()

    hist, bins = np.histogram(t_val, bins=100)
    center = (bins[:-1] + bins[1:]) / 2
    width = 2 * (bins[1] - bins[0])
    fig = plt.figure()
    fig.suptitle('Throttle Distribution', fontsize=20)
    plt.xlabel('throttle values', fontsize=18)
    plt.ylabel('Frequency', fontsize=16)
    plt.bar(center, hist, align='center', width = width)
    plt.show()

    '''#show speed where throttle is the lowest
    lowest_t_dataframe = data_frame.nsmallest(5, 'throttle')
    for index, row in lowest_t_dataframe.iterrows():
        img = row.iloc[0]
        image = cv2.imread('data/'+img)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        #print(img)'''

    '''#show speed where throttle is the highest
    lowest_t_dataframe = data_frame.nlargest(5, 'throttle')
    for index, row in lowest_t_dataframe.iterrows():
        img = row.iloc[0]
        image = cv2.imread('data/'+img)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        #print(img)'''

def get_data_stats(data_frame):
    '''get data stastics like speed distribution, steering angle distribution,
    throttle distribution etc'''
    print('Data statistics')

    print('Distribution of data with respect to steering angles')
    get_steering_angle_stats(data_frame)

    print('Distribution of data with respect to car speeds')
    get_speed_stats(data_frame)

    print('Distribution of data with respect to car throttle')
    get_throttle_stats(data_frame)


data_frame = load_data()
get_data_stats(data_frame)
train, valid, data_frame = filter_dataset(data_frame)
get_data_stats(data_frame)

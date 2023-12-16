import cv2 as cv
import os
import random
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')

bicycle = config.get('size_paths','Bicycle_path')
motorcycle = config.get('size_paths', 'Motorcycle_path')
car = config.get('size_paths', 'Car_path')
bus = config.get('size_paths', 'Bus_path')
commercial = config.get('size_paths', 'Commercial_path')
input_size = config.get('size_paths', 'input_size')

def resized(image_path,image):

    '''
    image_path for getting basename of image so while rewriting use the same name as that of read
    image: pass read image and resize based on video captured or resolution of video
    '''

    image_basename = os.path.basename(image_path) # gives the name of image along with extension
    basename = os.path.splitext(image_basename)[0] # gives only the name of image without extension 
    
    aspect_ratio = (image.shape[1] / image.shape[0]) # get the aspect ratio of image shape[1] = width, shape [0] = height
    desired_height = int(desired_width/aspect_ratio) # set height as per aspect ratio 
    resized_image = cv.resize(image, (desired_width, desired_height)) 
    
    cv.imwrite('./test/size/'+basename+'.png', resized_image)

    return resized_image

# Reference Image paths
Bicycle_path = bicycle
Motorcycle_path = motorcycle
Car_path = car
Bus_path = bus
Commercial_path = commercial

"""Bicycle_path = './Reference_Images/size/cycle.jpg'
Motorcycle_path = './Reference_Images/size/Bike.png'
Car_path = './Reference_Images/size/Car.jpg'
Bus_path = './Reference_Images/size/Bus.jpg'
Commercial_path = './Reference_Images/size/Commercial.jpg'"""

# Read Reference Images 
Bicycle = cv.imread(Bicycle_path)
Motorcycle = cv.imread(Motorcycle_path)
Car = cv.imread(Car_path)
Bus = cv.imread(Bus_path)
Commercial = cv.imread(Commercial_path)

path = input_size
#path = './Videos'
file = random.choice(os.listdir(path))
video = cv.VideoCapture(os.path.join(path, file))

if video.isOpened():
    video_width = video.get(cv.CAP_PROP_FRAME_WIDTH)
    video_height = video.get(cv.CAP_PROP_FRAME_HEIGHT)

desired_width = int(video_width)

# Resize images 
bicycle_resized_image = resized(Bicycle_path, Bicycle)
motorcycle_resized_image = resized(Motorcycle_path, Motorcycle)
Car_resized_image = resized(Car_path, Car)
Bus_resized_image = resized(Bus_path, Bus)
Rickshaw_resized_image = resized(Commercial_path, Commercial)

video.release()
cv.destroyAllWindows()
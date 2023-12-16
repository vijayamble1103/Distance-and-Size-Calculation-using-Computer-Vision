import cv2 as cv
import os
import time
import random
import torch 
import subprocess
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')

bicycle_distance_path = config.get('paths', 'Bicycle_distance_image')
motorcycle_distance_path = config.get('paths', 'Motorcycle_distance_image')
car_distance_path = config.get('paths', 'Car_distance_image')
bus_distance_path = config.get('paths', 'Bus_distance_image')
commercial_distance_path = config.get('paths', 'Commercial_distance_image')

bicycle_size_path = config.get('paths', 'Bicycle_size_image')
motorcycle_size_path = config.get('paths', 'Motorcycle_size_image')
car_size_path = config.get('paths', 'Car_size_image')
bus_size_path = config.get('paths', 'Bus_size_image')
commercial_size_path = config.get('paths', 'Commercial_size_image')

input_path = config.get('paths', 'input')

# Bounding Box Colors
green = (0,255,0)

# Text Color
yellow = (0,255,255)
white = (255,255,255)

# Text Background Color
peacock_green = (6,66,0)
candy = (43,31,198)
navy_blue = (128,0,0)
blue = (255,0,0)
sangria = (20,25,94)

start = time.time()
subprocess.run(['python', 'reference_image_converter_distance.py'])
subprocess.run(['python', 'reference_image_converter_size.py'])

def object_detection(image):
    results = model(image)
    extract_features = results.pandas().xyxy[0].to_dict(orient = 'records')

    data = []

    for features in extract_features:
        name = features['name']
        class_id = features['class']
        xmin = int(features['xmin'])
        xmax = int(features['xmax'])
        ymin = int(features['ymin'])
        ymax = int(features['ymax'])

        width = xmax - xmin

        if class_id == 1:
            data.append([name, class_id, width, (xmin, ymin - 5), (xmin, ymin - 20), xmin, ymin, xmax, ymax])
        elif class_id == 2:
            data.append([name, class_id, width, (xmin, ymin - 5), (xmin, ymin - 20), xmin, ymin, xmax, ymax])
        elif class_id == 3:
            data.append([name, class_id, width, (xmin, ymin - 5), (xmin, ymin - 20), xmin, ymin, xmax, ymax])
        elif class_id == 5:
            data.append([name, class_id, width, (xmin, ymin - 5), (xmin, ymin - 20), xmin, ymin, xmax, ymax])
        elif class_id == 7:
            data.append([name, class_id, width, (xmin, ymin - 5), (xmin, ymin - 20), xmin, ymin, xmax, ymax])
    
    return data

def focal_length_calculation(measured_distance, real_width, ref_width):
    '''
    Mathematical Calculation for finding the focal length of any object
    F = w * d / W
    F = Focal Length
    w = reference width i.e. width of an object in image
    d = image used for reference distance, for now we use object is 15 feet away from camera
    W = Real width i.e. width of an object in real 
    '''
    focal_length = (ref_width * measured_distance) / real_width #F = w * d / W 
    return focal_length

def distance_calculation(real_width, ref_width, focal_length):
    '''
    d = F * W / w
    '''
    measured_distance = (focal_length * real_width) / ref_width
    return measured_distance

def size_calculation(width_of_vehicle, distance_of_vehicle, focal_length):
    size = (width_of_vehicle * distance_of_vehicle) / focal_length 
    return size

def reference_image_focal_length(image):
    
    Reference_data = object_detection(image)
    Reference_object_width = Reference_data[0][2]

    return Reference_object_width 

def inches_to_feet(inches_value):
    conversion_inches_to_feet = inches_value * 0.0833 # 1 inch = 0.0833 feets
    return conversion_inches_to_feet

def pixel_to_feet(pixel_value):
    conversion_pixel_to_feet = pixel_value * 0.0104166667 # 1 pixel = 0.0104166667 feets
    return conversion_pixel_to_feet

def feet_to_inches(feet_value):
    conversion_feet_to_inches = feet_value * 12
    return conversion_feet_to_inches

def puttext():
    cv.rectangle(frame, (i[5], i[6]), (i[7], i[8]), green, 2)
    cv.putText(frame, f'Dist: {str(distance)} feets', i[3], cv.FONT_HERSHEY_COMPLEX, 0.5, yellow, 1)
    cv.putText(frame, f'Size: {str(size)} inches', i[4], cv.FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1)
    
choice = int(input("1. Bicycle \n2. Car \n3. Motorcycle \n4. Bus \n5. Commercial \n6. All \nEnter your choice: "))

Known_distance = 180 #inches
Bicycle_width = 30 #inches
Motorcycle_width = 31 #inches
Car_width = 70 #inches
Bus_width = 94 #inches
Rickshaw_width = 47 #inches

model = torch.hub.load('ultralytics/yolov5','yolov5s')

# 1. bicycle, 2. car, 3. motorcycle, 5. bus, and 7. commercial 
model.classes = [1, 2, 3, 5, 7]

# Read Reference Images for Distance calculation 
Bicycle_distance_image = cv.imread(bicycle_distance_path)
Motorcycle_distance_image = cv.imread(motorcycle_distance_path)
Car_distance_image = cv.imread(car_distance_path)
Bus_distance_image = cv.imread(bus_distance_path)
Commercial_distance_image = cv.imread(commercial_distance_path)

# Read Reference Images for Size calculation
Bicycle_size_image = cv.imread(bicycle_size_path)
Motorcycle_size_image = cv.imread(motorcycle_size_path)
Car_size_image = cv.imread(car_size_path)
Bus_size_image = cv.imread(bus_size_path)
Commercial_size_image = cv.imread(commercial_size_path)

# Focal length calculation for distance calculation
Bicycle_ref_width_distance = reference_image_focal_length(Bicycle_distance_image)
Bicycle_focal_length_distance = focal_length_calculation(Known_distance, Bicycle_width, Bicycle_ref_width_distance)

Car_ref_width_distance = reference_image_focal_length(Car_distance_image)
Car_focal_length_distance = focal_length_calculation(Known_distance, Car_width, Car_ref_width_distance)

Motorcycle_ref_width_distance = reference_image_focal_length(Motorcycle_distance_image)
Motorcycle_focal_length_distance = focal_length_calculation(Known_distance, Motorcycle_width, Motorcycle_ref_width_distance)

Bus_ref_width_distance = reference_image_focal_length(Bus_distance_image)
Bus_focal_length_distance = focal_length_calculation(Known_distance, Bus_width, Bus_ref_width_distance)

Rickshaw_ref_width_distance = reference_image_focal_length(Commercial_distance_image)
Rickshaw_focal_length_distance = focal_length_calculation(Known_distance, Rickshaw_width, Rickshaw_ref_width_distance)

# Focal length calculation for size calculation
Bicycle_ref_width_size = reference_image_focal_length(Bicycle_size_image)
Bicycle_focal_length_size = focal_length_calculation(Known_distance, Bicycle_width, Bicycle_ref_width_size)

Car_ref_width_size = reference_image_focal_length(Car_size_image)
Car_focal_length_size = focal_length_calculation(Known_distance, Car_width, Car_ref_width_size)

Motorcycle_ref_width_size = reference_image_focal_length(Motorcycle_size_image)
Motorcycle_focal_length_size = focal_length_calculation(Known_distance, Motorcycle_width, Motorcycle_ref_width_size)

Bus_ref_width_size = reference_image_focal_length(Bus_size_image)
Bus_focal_length_size = focal_length_calculation(Known_distance, Bus_width, Bus_ref_width_size)

Rickshaw_ref_width_size = reference_image_focal_length(Commercial_size_image)
Rickshaw_focal_length_size = focal_length_calculation(Known_distance, Rickshaw_width, Rickshaw_ref_width_size)

path = input_path
file = random.choice(os.listdir(path))
cap = cv.VideoCapture(os.path.join(path,file))
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv.CAP_PROP_FPS)
video = cv.VideoWriter('./Output.mp4', cv.VideoWriter_fourcc(*'MP42'),fps,(int(cap.get(3)),int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    test = object_detection(frame)
    for i in test:
        # BICYCLE
        if choice == 1 or choice == 6:
            if i[1] == 1:
                # Distance Calculation
                distance = distance_calculation(inches_to_feet(Bicycle_width), pixel_to_feet(i[2]), pixel_to_feet(Bicycle_focal_length_distance))
                distance = round(distance, 2)

                # Size Calculation
                focal_length_bicycle = focal_length_calculation(feet_to_inches(distance), Bicycle_width, Bicycle_ref_width_size)
                size = size_calculation(width, distance, focal_length_bicycle)
                #size = (width * distance) / focal_length_bicycle
                size = round(feet_to_inches(size), 2)

                # putText
                cv.rectangle(frame, (i[5], i[6]-35), (i[5]+155, i[6]), blue, -1)
                puttext()

                if choice == 1:
                    cv.rectangle(frame, (int(width)-102,3), (int(width), 22), blue, -1)
                    cv.putText(frame, "BICYCLE", (int(width)-100, 20), cv.FONT_HERSHEY_COMPLEX, 0.7, white, 2)
                #print('Bicycle: ', distance, "Size: ",size)

        # CAR
        if choice == 2 or choice == 6:
            if i[1] == 2:
                # Distance Calculation
                distance = distance_calculation(inches_to_feet(Car_width), pixel_to_feet(i[2]), pixel_to_feet(Car_focal_length_distance))
                distance = round(distance, 2)

                # Size Calculation
                focal_length_car = focal_length_calculation(feet_to_inches(distance), Car_width, Car_ref_width_size)
                size = size_calculation(width, distance, focal_length_car)
                #size = (width * distance) / focal_length_car 
                size = round(feet_to_inches(size), 2)

                # putText
                cv.rectangle(frame, (i[5], i[6]-35), (i[5]+155, i[6]), navy_blue, -1)
                puttext()

                if choice == 2:
                    cv.rectangle(frame, (int(width)-52,3), (int(width), 22), navy_blue, -1)
                    cv.putText(frame, "CAR", (int(width)-50, 20), cv.FONT_HERSHEY_COMPLEX, 0.7, white, 2)
                #print('Car: ',distance,"Size: ",size)
            
        # MOTORCYCLE
        if choice == 3 or choice == 6:
            if i[1] == 3:
                # Distance Calculation
                distance = distance_calculation(inches_to_feet(Motorcycle_width), pixel_to_feet(i[2]), pixel_to_feet(Motorcycle_focal_length_distance))
                distance = round(distance, 2)

                # Size Calculation
                focal_length_motorcycle = focal_length_calculation(feet_to_inches(distance), Motorcycle_width, Motorcycle_ref_width_size)
                size = size_calculation(width, distance, focal_length_motorcycle)
                size = round(feet_to_inches(size), 2)

                # putText
                cv.rectangle(frame, (i[5], i[6]-35), (i[5]+155, i[6]), peacock_green, -1)
                puttext()

                if choice == 3:
                    cv.rectangle(frame, (int(width)-152,3), (int(width), 22), peacock_green, -1)
                    cv.putText(frame, "MOTORCYCLE", (int(width)-150, 20), cv.FONT_HERSHEY_COMPLEX, 0.7, white, 2)
                #print('Motorcycle: ',distance,"Size: ",size)
        
        # BUS
        if choice == 4 or choice == 6:
            if i[1] == 5:
                # Distance Calculation
                distance = distance_calculation(inches_to_feet(Bus_width), pixel_to_feet(i[2]), pixel_to_feet(Bus_focal_length_distance))
                distance = round(distance, 2)

                # Size Calculation
                focal_length_bus = focal_length_calculation(feet_to_inches(distance), Bus_width, Bus_ref_width_size)
                size = size_calculation(width, distance, focal_length_bus)
                size = round(feet_to_inches(size), 2)

                # putText
                cv.rectangle(frame, (i[5], i[6]-35), (i[5]+155, i[6]), sangria, -1)
                puttext()

                if choice == 4:
                    cv.rectangle(frame, (int(width)-52,3), (int(width), 22), sangria, -1)
                    cv.putText(frame, "BUS", (int(width)-50, 20), cv.FONT_HERSHEY_COMPLEX, 0.7, white, 2)
                #print('Bus: ',distance,"Size: ",size)

        # COMMERCIAL
        if choice == 5 or choice == 6:
            if i[1] == 7:
                # Distance Calculation
                distance = distance_calculation(inches_to_feet(Rickshaw_width), pixel_to_feet(i[2]), pixel_to_feet(Rickshaw_focal_length_distance))
                distance = round(distance, 2)

                # Size Calculation
                focal_length_rickshaw = focal_length_calculation(feet_to_inches(distance), Rickshaw_width, Rickshaw_ref_width_size)
                size = size_calculation(width, distance, focal_length_rickshaw)
                size = round(feet_to_inches(size), 2)

                # putText
                cv.rectangle(frame, (i[5], i[6]-35), (i[5]+155, i[6]), candy, -1)
                puttext()

                if choice == 5:
                    cv.rectangle(frame, (int(width)-152,3), (int(width), 22), candy, -1)
                    cv.putText(frame, "COMMERCIAL", (int(width)-150, 20), cv.FONT_HERSHEY_COMPLEX, 0.7, white, 2)
                #print('Commercial: ',distance,"Size: ",size)

        if choice == 6: 
            # Display CAR
            cv.rectangle(frame, (int(width)-52,3), (int(width), 22), navy_blue, -1) # Background color for text
            cv.rectangle(frame, (int(width)-70,3), (int(width)-60, 22), navy_blue, -1) # Legend color
            cv.putText(frame, "CAR", (int(width)-50, 20), cv.FONT_HERSHEY_COMPLEX, 0.7, white, 2)
            
            # Display BUS
            cv.rectangle(frame, (int(width)-52,24), (int(width), 46), sangria, -1) # Background color for text
            cv.rectangle(frame, (int(width)-70,24), (int(width)-60, 46), sangria, -1) # Legend color
            cv.putText(frame, "BUS", (int(width)-50, 44), cv.FONT_HERSHEY_COMPLEX, 0.7, white, 2)
            
            # Display BICYCLE
            cv.rectangle(frame, (int(width)-102,48), (int(width), 68), blue, -1) # Background color for text
            cv.rectangle(frame, (int(width)-120,48), (int(width)-110, 68), blue, -1) # Legend color
            cv.putText(frame, "BICYCLE", (int(width)-100, 66), cv.FONT_HERSHEY_COMPLEX, 0.7, white, 2)

            # Display COMMERCIAL
            cv.rectangle(frame, (int(width)-152,70), (int(width), 90), candy, -1) # Background color for text
            cv.rectangle(frame, (int(width)-170,70), (int(width)-160, 90), candy, -1) # Legend color
            cv.putText(frame, "COMMERCIAL", (int(width)-150, 88), cv.FONT_HERSHEY_COMPLEX, 0.7, white, 2)
            
            # Display MOTORCYCLE
            cv.rectangle(frame, (int(width)-152,92), (int(width), 114), peacock_green, -1) # Background color for text
            cv.rectangle(frame, (int(width)-170,92), (int(width)-160, 114), peacock_green, -1) # Legend color
            cv.putText(frame, "MOTORCYCLE", (int(width)-150, 112), cv.FONT_HERSHEY_COMPLEX, 0.7, white, 2)

    cv.imshow("Output", frame)

    video.write(frame)
    if cv.waitKey(0) & 0xFF == ord('q'):
        time = time.time() - start
        print("="*100)
        print("Time taken for execution : ", round(time, 2), " seconds")
        break
cap.release()
cv.destroyAllWindows()




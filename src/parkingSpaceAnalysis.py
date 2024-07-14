# importing libraries
import cv2
import random
import numpy as np
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize YOLO model
model = YOLO('yolov8s.pt')

class ParkingSpace():
    def __init__(self):
        # Randomly selecting a image from the image dataset
        random_num = random.randint(1, 64)
        path = "Datasets/parkingLotImages/ezgif-frame-0" + str(random_num) + ".jpg"
        self.image_path = path
        #read csv file
        self.dataset = 'Datasets/parking_lot.csv'

    def parkingLotImg(self):

        # Reading the frame using OpenCV
        frame = cv2.imread(self.image_path)

        # Resize the image if needed
        frame = cv2.resize(frame, (1020, 500))

        # Object detection
        results = model.predict(frame)

        # Defining parking areas
        area1=[(52,364),(30,417),(73,412),(88,369)]

        area2=[(105,353),(86,428),(137,427),(146,358)]

        area3=[(159,354),(150,427),(204,425),(203,353)]

        area4=[(217,352),(219,422),(273,418),(261,347)]

        area5=[(274,345),(286,417),(338,415),(321,345)]

        area6=[(336,343),(357,410),(409,408),(382,340)]

        area7=[(396,338),(426,404),(479,399),(439,334)]

        area8=[(458,333),(494,397),(543,390),(495,330)]

        area9=[(511,327),(557,388),(603,383),(549,324)]

        area10=[(564,323),(615,381),(654,372),(596,315)]

        area11=[(616,316),(666,369),(703,363),(642,312)]

        area12=[(674,311),(730,360),(764,355),(707,308)]

        nonEmptySlots = 0
        # Process detected objects
        for result in results:
            boxes = result.boxes # Accessing bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = box.cls
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name = result.names[int(class_id)]

                if 'car' in class_name:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # Check if the center of the car is inside each parking slot/area
                    results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False)
                    if results1 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1
                    
                    results2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False)
                    if results2 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1

                    results3 = cv2.pointPolygonTest(np.array(area3, np.int32), (cx, cy), False)
                    if results3 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1
                    
                    results4 = cv2.pointPolygonTest(np.array(area4, np.int32), (cx, cy), False)
                    if results4 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1
                    
                    results5 = cv2.pointPolygonTest(np.array(area5, np.int32), (cx, cy), False)
                    if results5 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1
                    
                    results6 = cv2.pointPolygonTest(np.array(area6, np.int32), (cx, cy), False)
                    if results6 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1
                    
                    results7 = cv2.pointPolygonTest(np.array(area7, np.int32), (cx, cy), False)
                    if results7 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1
                    
                    results8 = cv2.pointPolygonTest(np.array(area8, np.int32), (cx, cy), False)
                    if results8 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1
                    
                    results9 = cv2.pointPolygonTest(np.array(area9, np.int32), (cx, cy), False)
                    if results9 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1
                    
                    results10 = cv2.pointPolygonTest(np.array(area10, np.int32), (cx, cy), False)
                    if results10 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1

                    
                    results11 = cv2.pointPolygonTest(np.array(area11, np.int32), (cx, cy), False)
                    if results11 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1

                    
                    results12 = cv2.pointPolygonTest(np.array(area12, np.int32), (cx, cy), False)
                    if results12 >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        cv2.putText(frame, str(class_name), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        nonEmptySlots += 1

        # available slots in the parking lot
        emptySlots = 12 - nonEmptySlots

        # Display the results
        print("Available slots for parking:" + str(emptySlots))
        cv2.namedWindow('Parking Lot')
        cv2.putText(frame,str(emptySlots),(23,30),cv2.FONT_HERSHEY_PLAIN,3,(127,127,127),2)
        cv2.imshow("Parking Lot", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def slotAnalysis(self):
        df = pd.read_csv(self.dataset, parse_dates=['Entry_Time', 'Exit_Time'])

        # Convert Entry_Time and Exit_Time to datetime with specified format
        df['Entry_Time'] = pd.to_datetime(df['Entry_Time'], format='%H:%M')
        df['Exit_Time'] = pd.to_datetime(df['Exit_Time'], format='%H:%M')

        # Aggregate parking occupancy by slot and hour
        df['entry_hour'] = df['Entry_Time'].dt.hour
        df['exit_hour'] = df['Exit_Time'].dt.hour

        # Create a DataFrame to store hourly occupancy for each slot
        slot_hourly_occupancy = pd.DataFrame(index=range(24))

        # Iterate over each slot and count hourly occupancy
        for slot_id, slot_data in df.groupby('Slot'):
            hourly_counts = slot_data.apply(
                lambda row: pd.Series(range(row['entry_hour'], row['exit_hour']+1)), axis=1
            ).stack().value_counts().reindex(range(24), fill_value=0)
            
            slot_hourly_occupancy[f'Slot {slot_id}'] = hourly_counts

        # Plotting
        plt.figure(figsize=(12, 6))
        for slot_id in slot_hourly_occupancy.columns:
            plt.plot(slot_hourly_occupancy.index, slot_hourly_occupancy[slot_id], label=slot_id)

        plt.title('Hourly Occupancy for Each Parking Slot in a week')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Times Occupied')
        plt.xticks(range(24), [f'{hour}:00' for hour in range(24)])
        plt.grid(True)
        plt.legend(title='Slot')
        plt.tight_layout()
        plt.show()

        # Calculate the duration each slot is occupied
        df['Duration'] = (df['Exit_Time'] - df['Entry_Time']).dt.total_seconds() / 3600  # convert to hours

        # Calculate the average duration for each slot
        average_duration = df.groupby('Slot')['Duration'].mean()
        # print(average_duration)

        # Create a bar plot for the average duration
        plt.figure(figsize=(14, 8))
        sns.barplot(x=average_duration.index, y=average_duration.values, palette='viridis')

        # Set the labels and title
        plt.xlabel('Parking Slot')
        plt.ylabel('Average Occupancy Duration (hours)')
        plt.title('Average Occupancy Duration for Each Parking Slot')

        # Show the plot
        plt.show()
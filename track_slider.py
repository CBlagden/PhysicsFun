import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import numpy as np


def graph(x, y, title, xlabel='', ylabel=''):
        plt.figure(num=title)
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.margins(0)


def percent_error(experimental_val, true_val):
    return abs(experimental_val - true_val) / abs(true_val) * 100


class ObjectTracker:

    def __init__(self, file, lower_hsv_range, upper_hsv_range, 
        start_time=0, end_time=float('inf'), min_area=0):
        self.file = file

        self.lower_hsv_range = lower_hsv_range
        self.upper_hsv_range = upper_hsv_range

        self.start_time = start_time
        self.end_time = end_time

        self.min_area = min_area

    def process_video(self, show=False):
        cap = cv2.VideoCapture(self.file)
        start_time = time.time()

        times = []
        positions = []

        while cap.isOpened():
            if cv2.waitKey(25) & 0xFF == 'q':
                break

            ret, frame = cap.read()
            if ret is False:
                break

            img, cur_pos = self.process_img(frame)

            
            if show:
                cv2.imshow(self.file, img)

            rows, cols, _ = img.shape

            cur_time = time.time() - start_time

            if cur_time > self.end_time:
                break

            if cur_pos is not None and cur_time > self.start_time:
                cur_pos = self.pixels_to_meters(cur_pos)
                times.append(cur_time)
                positions.append(cur_pos)
                print("Time, {}  Position: {}".format(cur_time, cur_pos))

        if show:
            cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
        return times, positions

    def process_img(self, img):
        
        img_resized = self.resize_img(img)

        img_hsv = cv2.cvtColor(img_resized, 
            cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(img_hsv, 
            self.lower_hsv_range, 
            self.upper_hsv_range)
        mask = cv2.erode(mask, None, iterations=4)
        mask = cv2.dilate(mask, None, iterations=4)

        cv2.imshow('mask', mask)

        _, contours, _ = cv2.findContours(mask, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE)
        x = None
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            area = abs(w*h)
            if area > self.min_area:
                cv2.rectangle(img_resized, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)
            else:
                x = None

        return img_resized, x

    def resize_img(self, img):
        pass

    def pixels_to_meters(self, pixels):
        return pixels * 2.1 / 500.0

    def get_new_pos_vel_and_accel(self, times, positions, k=2):

        times = np.array(times) * 0.191

        pos_poly = np.poly1d(np.polyfit(times, positions, deg=k))
        vel_poly = np.polyder(pos_poly)
        acc_poly = np.polyder(pos_poly, m=2)

        return pos_poly(times), vel_poly(times), acc_poly(times), times


class SlowSlider(ObjectTracker):

    def __init__(self, file):
        super().__init__(file, start_time=33.0, end_time=40.0, 
            lower_hsv_range=np.array([0.0, 0.0, 151.0]),
            upper_hsv_range=np.array([188.0, 25.0, 255.0]),
            min_area
=8.0)

    def resize_img(self, img):
        rows, cols, _ = img.shape
        return img[int(0.3*rows):int(0.7*rows), int(0.04*cols):]

class FastSlider(ObjectTracker):

    def __init__(self, file):
        super().__init__(file, start_time=9.5, end_time=17.1, 
            lower_hsv_range=np.array([0.0, 0.0, 175.0]),
            upper_hsv_range=np.array([255.0, 16.0, 246.0]),
            min_area
=8.0)

    def resize_img(self, img):
        rows, cols, _ = img.shape
        return img[int(0.48*rows):int(0.62*rows), int(0.0*cols):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True, help="path to the video file")
    parser.add_argument('-s', '--speed', required=True, help='slow or fast acceleration')

    args = vars(parser.parse_args())
    tracker = None
    
    if args['speed'] in ('fast', 'f'):
        tracker = FastSlider(args['video'])
    elif args['speed'] in ('slow', 's'):
        tracker = SlowSlider(args['video'])
    else:
        tracker = FastSlider(args['video'])
   
    times, positions = tracker.process_video(show=True)
    pos, vel, accel, times = tracker.get_new_pos_vel_and_accel(times, positions)
    graph(times, accel, 'acceleration', xlabel='Time (s)', ylabel='Acceleration (m/s^2)')
    print("Acceleration: %f" % accel[0])
    plt.show()
    print("Error: %f" % percent_error(accel[0], 1.05))

if __name__ == '__main__':
    main()
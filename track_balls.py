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
        start_time=0, end_time=float('inf'), min_radius=0):
        self.file = file

        self.lower_hsv_range = lower_hsv_range
        self.upper_hsv_range = upper_hsv_range

        self.start_time = start_time
        self.end_time = end_time

        self.min_radius = min_radius

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
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > self.min_radius:
                cv2.circle(img_resized,
                 (int(x), int(y)), 
                    int(radius), (0, 255, 255), 2)    
            else:
                x = None

        return img_resized, x

    def resize_img(self, img):
        pass

    def pixels_to_meters(self, pixels):
        return pixels * 2.1 / 500.0

    def get_new_pos_vel_and_accel(self, times, positions, k=2):

        times = np.array(times) * 0.168

        pos_poly = np.poly1d(np.polyfit(times, positions, deg=k))
        vel_poly = np.polyder(pos_poly)
        acc_poly = np.polyder(pos_poly, m=2)

        return pos_poly(times), vel_poly(times), acc_poly(times), times


class TennisBallTracker(ObjectTracker):

    def __init__(self, file):
        super().__init__(file, start_time=4.8, end_time=10.2,
            lower_hsv_range=np.array([28.0, 89.0, 45.0]),
            upper_hsv_range=np.array([154.0, 187.0, 170.0]),
            min_radius=8.0)

    def resize_img(self, img):
        rows, cols, _ = img.shape
        return img[int(0.42*rows):int(0.67*rows), int(0.04*cols):]

class BaseBallTracker(ObjectTracker):

    def __init__(self, file):
        super().__init__(file, start_time=12.5, end_time=20.3, 
            lower_hsv_range=np.array([13.0, 64.0, 19.0]),
            upper_hsv_range=np.array([122.0, 120.0, 255.0]),
            min_radius=8.0)

    def resize_img(self, img):
        rows, cols, _ = img.shape
        return img[int(0.5*rows):int(0.67*rows), int(0.04*cols):]

class TennisBallUpAndDown(ObjectTracker):

    def __init__(self, file):
        super().__init__(file, 
            start_time=2.3,
            end_time=10.0,
            lower_hsv_range=np.array([28.0, 89.0, 45.0]),
            upper_hsv_range=np.array([154.0, 187.0, 255.0]),
            min_radius=8.0)

    def resize_img(self, img):
        rows, cols, _ = img.shape
        return img[int(0.41*rows):int(0.67*rows), int(0.04*cols):int(0.43*cols)]


class TennisBallWithBaseBallTracker(ObjectTracker):

    def __init__(self, file):
        super().__init__(file, start_time=12.5, end_time=16.1, 
            lower_hsv_range=np.array([28.0, 89.0, 45.0]),
            upper_hsv_range=np.array([154.0, 187.0, 255.0]),
            min_radius=8.0)

    def resize_img(self, img):
        rows, cols, _ = img.shape
        return img[int(0.4*rows):int(0.6*rows), int(0.04*cols):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required=True, help="path to the video file")
    parser.add_argument('-b', '--ball', required=True, help='ball in video: baseball or tennisball')

    args = vars(parser.parse_args())
    tracker = None
    
    if args['ball'] in ('baseball', 'b'):
        tracker = BaseBallTracker(args['video'])
    elif args['ball'] in ('tennisball', 't'):
        tracker = TennisBallTracker(args['video'])
    elif args['ball'] in ('with', 'w'):
        tracker = TennisBallWithBaseBallTracker(args['video'])
    elif args['ball'] in ('upanddown' 'u'):
        tracker = TennisBallUpAndDown(args['video'])

    times, positions = tracker.process_video(show=True)
    pos, vel, accel, times = tracker.get_new_pos_vel_and_accel(times, positions)
    graph(times, pos, 'position', xlabel='Time (s)', ylabel='Position (m)')
    graph(times, vel, 'velocity', xlabel='Time (s)', ylabel='Velocity (m/s)')
    graph(times, accel, 'acceleration', xlabel='Time (s)', ylabel='Acceleration (m/s^2)')
    print("Percent error: %f" % percent_error(accel[0], 9.81))

    plt.show()

if __name__ == '__main__':
    main()
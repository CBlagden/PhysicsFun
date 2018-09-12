import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.interpolate

lower_hsv_range = np.array([1.0, 110.0, 100.0])
upper_hsv_range = np.array([15.0, 255, 150])

WIN_NAME = 'Car tracker'

def pixels_to_meters(pixels):
    return pixels / 130.0

def car_position(img):
    rows, cols, _ = img.shape
    img_resized = img[int(0.5 * rows):int(0.7 * rows), int(0.02 * cols):]

    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(img_hsv, lower_hsv_range, upper_hsv_range)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = None
    max_area = 0
    for c in contours:
        contour_area = cv2.contourArea(c)
        if contour_area > max_area:
            max_area = contour_area
            max_contour = c

    x, y, w, h = cv2.boundingRect(max_contour)
    cv2.rectangle(img_resized, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)

    return img, x

def graph(times, poses):
    plt.figure(num='Racecar Phyics Lab')

    pos_spl = scipy.interpolate.UnivariateSpline(times, poses, k=1)
    plt.subplot(1, 2, 1)
    plt.plot(times, pos_spl(times))
    plt.title('Position (m) vs. Time (s)')

    vel_spl = pos_spl.derivative()
    plt.subplot(1, 2, 2)
    plt.title('Velocity(m/s) vs Time(s)')
    plt.plot(times, vel_spl(times))

    plt.ylim(ymin=0)

    plt.show()

def main():
    cap = cv2.VideoCapture('video/IMG_0972.3gp')
    start_time = time.time()
    times = []
    poses = []

    while cap.isOpened():
        if cv2.waitKey(25) & 0xFF == 'q':
            break

        ret, frame = cap.read()
        if ret is False:
            break

        img, pos = car_position(frame)
        rows, cols, _ = img.shape

        img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imshow(WIN_NAME, img)

        time_val = time.time() - start_time

        # filters waiting time and drop at end
        if time_val > 4.4:
            break
        elif time_val > 3.3:
            times.append(time_val)
            poses.append(pixels_to_meters(pos))

    graph(times=times, poses=poses)

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

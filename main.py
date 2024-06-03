import cv2
import numpy as np
import pyautogui
import time
import threading
import ctypes
from collections import deque


# Function to capture the screen in a specific region
def capture_screen(region):
    x, y, w, h = region
    screenshot = pyautogui.screenshot(region=(x, y, w, h))
    screen_np = np.array(screenshot)
    screen_cv2 = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
    return screen_cv2


# Function to simulate mouse click
def click_at(x, y):
    x = int(x)
    y = int(y)
    ctypes.windll.user32.SetCursorPos(x, y)
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP


# Function to apply non-maximum suppression to the template matching results
def non_max_suppression(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype("int")


# Function to find template and click
def find_and_click(capture_region, target_images, bomb_image, stop_event, scale_factors, click_deque):
    gray_bomb = cv2.cvtColor(bomb_image, cv2.COLOR_BGR2GRAY)
    bomb_w, bomb_h = gray_bomb.shape[::-1]

    while not stop_event.is_set():
        screen = capture_screen(capture_region)
        if screen is None:
            time.sleep(0.05)
            continue

        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        bomb_locations = []

        for scale in scale_factors:
            resized_bomb = cv2.resize(gray_bomb, (int(bomb_w * scale), int(bomb_h * scale)))
            result_bomb = cv2.matchTemplate(gray_screen, resized_bomb, cv2.TM_CCOEFF_NORMED)
            loc_bomb = np.where(result_bomb >= 0.8)

            for pt in zip(*loc_bomb[::-1]):
                bomb_locations.append([pt[0], pt[1], pt[0] + resized_bomb.shape[1], pt[1] + resized_bomb.shape[0]])

        bomb_locations = np.array(bomb_locations)
        bomb_locations = non_max_suppression(bomb_locations, 0.3)

        for target_image in target_images:
            gray_template = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
            template_w, template_h = gray_template.shape[::-1]
            target_locations = []

            for scale in scale_factors:
                resized_template = cv2.resize(gray_template, (int(template_w * scale), int(template_h * scale)))
                result = cv2.matchTemplate(gray_screen, resized_template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(result >= 0.8)

                for pt in zip(*loc[::-1]):
                    target_locations.append(
                        [pt[0], pt[1], pt[0] + resized_template.shape[1], pt[1] + resized_template.shape[0]])

            target_locations = np.array(target_locations)
            target_locations = non_max_suppression(target_locations, 0.3)

            for (x1, y1, x2, y2) in target_locations:
                click_point = (x1 + capture_region[0] + (x2 - x1) // 2,
                               y1 + capture_region[1] + (y2 - y1) // 2)

                if click_point not in click_deque:
                    click_deque.append(click_point)
                    if len(click_deque) > 100:
                        click_deque.popleft()

                    bomb_overlap = False
                    for (bx1, by1, bx2, by2) in bomb_locations:
                        if (bx1 < x1 < bx2 or bx1 < x2 < bx2) and (by1 < y1 < by2 or by1 < y2 < by2):
                            bomb_overlap = True
                            break

                    if not bomb_overlap:
                        click_at(*click_point)
                        time.sleep(0.01)

        time.sleep(0.05)


def main():
    try:
        coin_path = "coins.png"

        ice_path = "ice.png"
        bomb_path = "bomb.png"

        capture_region = (770, 222, 375, 605)
        x, y, w, h = capture_region

        # Define four subregions
        regions = [
            (x, y, w // 2, h // 2),  # Top-left
            (x + w // 2, y, w // 2, h // 2),  # Top-right
            (x, y + h // 2, w // 2, h // 2),  # Bottom-left
            (x + w // 2, y + h // 2, w // 2, h // 2)  # Bottom-right
        ]

        coin_image = cv2.imread(coin_path, cv2.IMREAD_COLOR)
        ice_image = cv2.imread(ice_path, cv2.IMREAD_COLOR)
        bomb_image = cv2.imread(bomb_path, cv2.IMREAD_COLOR)

        if coin_image is None or ice_image is None or bomb_image is None:
            print("Failed to load template images.")
            return

        stop_event = threading.Event()
        click_deque = deque(maxlen=100)
        scale_factors = [1.0, 0.9, 0.8, 1.1]  # Add scale factors for multi-scale matching

        target_images = [coin_image, ice_image]

        # Create threads for each region
        threads = []
        for region in regions:
            threads.append(threading.Thread(target=find_and_click, args=(
            region, target_images, bomb_image, stop_event, scale_factors, click_deque)))

        # Start all threads
        for thread in threads:
            thread.start()

        print("Press Enter to stop...")
        input()

        stop_event.set()

        # Join all threads
        for thread in threads:
            thread.join()
    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    main()

import cv2
import re

class Board:
    """
    this class implement the grid world problem as a frozen lake problem.
    """
    def __init__(self, resolution):
        self.current_full_img = None
        self.current_full_bboxes = None
        self.current_img = None
        self.current_bboxes = None
        self.zoom_step = 0
        self.move_step = 0
        self.resolution = resolution

    def init_env(self, image):
        self.current_full_img = cv2.imread(image)


    def get_reward(self):
        pass

    def go_to_next_state(self):
        pass

    def zoom(self):
        self.zoom_step += 1

    def move(self):
        pass

    def check_cuda(self):
        if __name__ == '__main__':
            cv_info = [re.sub('\s+', ' ', ci.strip()) for ci in cv2.getBuildInformation().strip().split('\n')
                       if len(ci) > 0 and re.search(r'(nvidia*:?)|(cuda*:)|(cudnn*:)', ci.lower()) is not None]
            return len(cv_info) > 0


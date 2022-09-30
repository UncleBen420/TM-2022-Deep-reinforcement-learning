from time import sleep

import matplotlib.pyplot as plt

from EnvironmentSoft import SoftEnv

if __name__ == '__main__':
    se = SoftEnv(200, 1, 1)

    se.init_env("/home/remy/Documents/P9467.png", "/home/remy/Documents/P9467.txt")
    se.x = 3
    se.y = 3
    se.compute_sub_img()
    plt.imshow(se.full_img)
    plt.show()
    plt.imshow(se.sub_img)
    plt.show()
    plt.imshow(se.marked_image)
    plt.show()
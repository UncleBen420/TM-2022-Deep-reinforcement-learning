from matplotlib import pyplot as plt

from BasicImplementation.Agents import QLearning, E_Greedy
from BasicImplementation.EnvironmentSoft import SoftEnv

if __name__ == '__main__':
    se = SoftEnv(200, "./MobileNetV3/Models/boat_model.pth", "./MobileNetV3/Models/surface_model.pth")

    se.init_env("/home/remy/Documents/P9467.png", "/home/remy/Documents/P9467.txt")
    ql = QLearning(se, E_Greedy(0.1), episodes=10)
    ql.fit()

    plt.imshow(se.render_marked_map())
    plt.imshow()

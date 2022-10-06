# This is a sample Python script.
import imageio
import matplotlib.pyplot as plt

from Agents.TD import QLearning, E_Greedy
from Environment.dummy import DummyEnv
from Environment.environment import SoftEnv
from Model.dummy import DummyNET
from Model.model import MobileRLNET

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    se = DummyEnv()
    se.init_env()
    ql = QLearning(se, DummyNET(learning_rate=0.01), E_Greedy(0.1), episodes=300)
    loss, v, rewards, boats_left = ql.fit()

    plt.plot(loss)
    plt.show()
    plt.plot(v)
    plt.show()
    plt.plot(rewards)
    plt.show()
    plt.plot(boats_left)
    plt.show()

    plt.imshow(se.render_board_img(se.grid, se.marked_map, [1, 0, 0]))
    plt.show()

    imageio.mimsave("haha.gif", se.evolution, duration=0.05)
    #se.get_gif_trajectory("dummy_deep.gif")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

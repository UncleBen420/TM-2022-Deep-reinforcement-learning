# This is a sample Python script.
from Agents.TD import QLearning
from Environment.environment import SoftEnv
from Model.model import MobileRLNET

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    se = SoftEnv(224)
    se.init_env("/home/remy/Documents/P9467.png", "/home/remy/Documents/P9467.txt")
    ql = QLearning(se, MobileRLNET(), E_Greedy(0.1), episodes=10)
    ql.fit()
    QLearning()
    print(mrln.mb_net)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

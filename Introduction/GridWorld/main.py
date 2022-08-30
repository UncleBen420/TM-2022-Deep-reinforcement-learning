from DynamicProgramming.DP import DP
from Environment.GridWorld import Board

import matplotlib.pyplot as plt

if __name__ == '__main__':
    board = Board(nb_trap=30, size=10)
    print(board.render_board())

    dp = DP(board, threshold=0.0001, gamma=0.5)
    history, v = dp.policy_iteration()

    for policy in history:
        rendered = board.render_policy(policy)
        print(rendered)

    plt.plot(v)
    plt.show()

import itertools
from pyboy import WindowEvent
import numpy as np

def GetActions():
    baseActions = [WindowEvent.PRESS_BUTTON_A,
                    WindowEvent.PRESS_BUTTON_B,
                    WindowEvent.PRESS_ARROW_UP,
                    WindowEvent.PRESS_ARROW_DOWN,
                    WindowEvent.PRESS_ARROW_LEFT,
                    WindowEvent.PRESS_ARROW_RIGHT
                    ]

    totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
    withoutRepeats = []

    for combination in totalActionsWithRepeats:
        reversedCombination = combination[::-1]
        if (reversedCombination not in withoutRepeats):
            withoutRepeats.append(combination)

    filteredActions = [[action] for action in baseActions] + withoutRepeats

    return filteredActions

# print(GetActions())
x = [1, 2]
# choice() chooses an index, 0 or 1
print(np.random.choice(len(x)))
import random
import math
from typing import List, Tuple

# Rectified Linear Unit activation function
def ReLU(i: float) -> float:
    return max(0, i)

# d/dx ReLU(x)
# We define the derivative to be 1 at x=0, despite the fact that ReLU's derivative
# is not defined at x = 0. I do not actually know why we are allowed to do this.
# To any calculus wizards reading this, please feel free to explain why this is OK.
def ReLUDerivative(x: float) -> float:
    if x >= 0:
        return 1
    return 0

class TwoParameterReLUNetwork:

    def __init__(self) -> None:
        self.weight = random.uniform(0, 1)
        self.bias = random.uniform(0, 1)
    
    def FeedForward(self, input: float) -> float:
        return ReLU(self.weight * input + self.bias)
    
    def Show(self) -> None:
        print("weight:", str(self.weight))
        print("bias:", str(self.bias))

class TwoParameterReLUNetworkTrainer:
    
    def __init__(self, tprn: TwoParameterReLUNetwork, trainingData: List[Tuple[float, float]]) -> None:
        self.tprn = tprn
        self.trainingData = trainingData
    
    def CalculateNetworkCost(self) -> float:
		# <Exercise>
        # Write a function to compute the cost of the network for arbitrary numbers of training datapoints
        costs = map(lambda t: math.pow(tprn.FeedForward(t[0]) - t[1], 2), trainingData)
        return sum(costs) / len(trainingData)

    def CalculateGradient(self) -> Tuple[float, float]:
        # <Exercise>
        # Write a function that computes the gradient for the parameters of the network given an arbitrary number of training
        # examples. The final gradient should be -1 times the average of the gradients for all of the training examples.
        bs = map(lambda t: 2 * (tprn.FeedForward(t[0]) - t[1]) * ReLUDerivative(tprn.weight * t[0] + tprn.bias), trainingData)
        ws = map(lambda t: 2 * math.pow(t[0], 2) * (tprn.FeedForward(t[0]) - t[1]) * ReLUDerivative(tprn.weight * t[0] + tprn.bias), trainingData)
        gb = sum(bs) / len(trainingData)
        gw = sum(ws) / len(trainingData)
        return (-gw, -gb)

    def TrainNetwork(self, learningRate: float) -> None:
        count = 0
        while True:
            (gw, gb) = self.CalculateGradient()
            if math.sqrt(gw**2 + gb**2) > 0.0001:
                tprn.weight += gw * learningRate
                tprn.bias += gb * learningRate
                count += 1
            else:
                print("solved in", str(count), "steps")
                break

trainingData: List[Tuple[float, float]] = [
    (0, 2),
    (2, 3),
]

tprn = TwoParameterReLUNetwork()
tprnt = TwoParameterReLUNetworkTrainer(tprn, trainingData)

tprn.Show()

tprnt.TrainNetwork(0.01)

tprn.Show()
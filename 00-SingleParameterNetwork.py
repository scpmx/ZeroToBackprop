import random

class SingleParameterNetwork:

    def __init__(self, w: float) -> None:
        self.w = w
    
    def FeedForward(self, i: float) -> float:
        # <Exercise>
        # Apply the network to the input i
        return self.w * i

    def ApplyGradient(self, grad: float, learningRate: float) -> None:
        # <Exercise>
        # Apply the gradient to the param proportional to the learning rate
        self.w += grad * learningRate

    def GetParams(self) -> float:
        return self.w
    
    def Show(self) -> None:
        print("weight:", str(self.w))

class SingleParameterNetworkTrainer:

    def __init__(self, ti: float, ta: float) -> None:
        self.ti = ti
        self.ta = ta
    
    def ComputeNetworkCost(self, spn: SingleParameterNetwork) -> float:
        # <Exercise>
        # Use mean squared error to calculate a cost of the network relative
        # to the training data 
        actual = spn.FeedForward(self.ti)
        mean = actual - self.ta
        return pow(mean, 2)

    def ComputeGradient(self, spn: SingleParameterNetwork) -> float:
        # <Exercise>
        # Use the chain rule to find dC/dw 
        return (2 * spn.GetParams() * pow(self.ti, 2)) - (2 * self.ti * self.ta)

    def Fit(self, spn: SingleParameterNetwork, learningRate: float) -> None:
        # <Exercise>
        # Write a basic gradient descent algorithm 
        grad = self.ComputeGradient(spn)
        while grad > 0.00001:
            spn.ApplyGradient(-1 * grad, learningRate)
            grad = self.ComputeGradient(spn)

# Randomly initialize parameter for network
w = random.uniform(0, 1)
spn = SingleParameterNetwork(w)

# Randomly initialize training input and output
ti = random.uniform(0.1, 1)
ta = random.uniform(0, 1)

# Print initial values to screen
print("ti:", str(ti), "ta:", str(ta))
spn.Show()

# Initialize a network trainer and use it to fit our single parameter
# network to our training data with a learning rate of 0.1
spnt = SingleParameterNetworkTrainer(ti, ta)
spnt.Fit(spn, 0.1)

# Print final network cost
finalCost = spnt.ComputeNetworkCost(spn)
print("final cost:", str(finalCost))
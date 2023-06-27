import random
import math
from typing import Tuple

class TwoParameterLinearNetwork:
	
	def __init__(self) -> None:
		self.weight = random.uniform(0, 1)
		self.bias = random.uniform(0, 1)
		
	def FeedForward(self, input: float) -> float:
		# <Exercise>
		# Apply the network to the input i
		return input * self.weight + self.bias
	
	def ApplyGradient(self, wGradient: float, bGradient: float, learningRate: float) -> None:
		# <Exercise>
		# Apply the gradient to the params proportional to the learning rate
		self.weight += wGradient * learningRate
		self.bias += bGradient * learningRate
		
	def Show(self) -> None:
		print("weight:", str(self.weight))
		print("bias:", str(self.bias))

class TwoParameterLinearNetworkTrainer:
	
	def __init__(
			self,
			tpln: TwoParameterLinearNetwork,
			t00: float,
			t01: float,
			t10: float,
			t11: float,
			learningRate: float) -> None:
		self.tpln = tpln
		self.t00 = t00
		self.t01 = t01
		self.t10 = t10
		self.t11 = t11
		self.learningRate = learningRate
		
	def CalculateNetworkCost(self) -> float:
		c0 = pow(tpln.FeedForward(t00) - t01, 2)
		c1 = pow(tpln.FeedForward(t10) - t11, 2)
		return (c0 + c1) / 2
	
	def CalculateGradient(self) -> Tuple[float, float]:
		w = tpln.weight
		b = tpln.bias
		gw0 = 2 * (w * pow(t00, 2) + (b * t00) - (t00 * t01))
		gw1 = 2 * (w * pow(t10, 2) + (b * t10) - (t10 * t11))
		gw = (gw0 + gw1) / 2
		gb0 = 2 * (w * t00 + b - t01)
		gb1 = 2 * (w * t10 + b - t11)
		gb = (gb0 + gb1) / 2
		return (-gw, -gb)
	
	def TrainNetwork(self) -> None:
		cnt = 0
		(gw, gb) = self.CalculateGradient()
		while math.sqrt(gw*gw+gb*gb) > 0.0001:
			tpln.ApplyGradient(gw, gb, self.learningRate)
			(gw, gb) = self.CalculateGradient()
			cnt += 1
		print("solved in", str(cnt), "steps")
			

tpln = TwoParameterLinearNetwork()
tpln.Show()

# training data
t00 = random.uniform(1,2)
t01 = random.uniform(1,2)
t10 = random.uniform(1,2)
t11 = random.uniform(1,2)
learningRate = 0.1

print(t00, t01, t10, t11)

tplnt = TwoParameterLinearNetworkTrainer(tpln, t00, t01, t10, t11, learningRate)

initialCost = tplnt.CalculateNetworkCost()
print("initial cost:", initialCost)

tplnt.TrainNetwork()
tpln.Show()

finalCost = tplnt.CalculateNetworkCost()
print("final cost:", finalCost)
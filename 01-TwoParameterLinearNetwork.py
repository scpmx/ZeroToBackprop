import random
import math

class TwoParameterLinearNetwork:
	
	def __init__(self, w: float, b: float):
		self.w = w
		self.b = b
		
	def FeedForward(self, i):
		return i * self.w + self.b
	
	def ApplyGradient(self, gw, gb, lr):
		self.w += gw * lr
		self.b += gb * lr
		
	def Show(self):
		print("w:", str(self.w))
		print("b:", str(self.b))

class TwoParameterLinearNetworkTrainer:
	
	def __init__(self, tpln, t00, t01, t10, t11, lr):
		self.tpln = tpln
		self.t00 = t00
		self.t01 = t01
		self.t10 = t10
		self.t11 = t11
		self.lr = lr
		
	def CalculateNetworkCost(self):
		c0 = pow(tpln.FeedForward(t00) - t01, 2)
		c1 = pow(tpln.FeedForward(t10) - t11, 2)
		return (c0 + c1) / 2
	
	def CalculateGradient(self):
		w = tpln.w
		b = tpln.b
		gw0 = 2 * (w * pow(t00, 2) + (b * t00) - (t00 * t01))
		gw1 = 2 * (w * pow(t10, 2) + (b * t10) - (t10 * t11))
		gw = (gw0 + gw1) / 2
		gb0 = 2 * (w * t00 + b - t01)
		gb1 = 2 * (w * t10 + b - t11)
		gb = (gb0 + gb1) / 2
		return (-gw, -gb)
	
	def FitNetwork(self):
		cnt = 0
		(gw, gb) = self.CalculateGradient()
		while math.sqrt(gw*gw+gb*gb) > 0.0001:
			tpln.ApplyGradient(gw, gb, self.lr)
			(gw, gb) = self.CalculateGradient()
			cnt += 1
		print("solved in", str(cnt), "steps")
			

w = random.uniform(0, 1)
b = random.uniform(0, 1)
tpln = TwoParameterLinearNetwork(w, b)

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

tplnt.FitNetwork()
tpln.Show()

finalCost = tplnt.CalculateNetworkCost()
print("final cost:", finalCost)
# This is a modified version of my original neural net class.  It has been generalized for 
# any number of output perceptrons.  This was a pretty straightforward process.  I simply 
# needed to replace my output perceptron object with an array of objects and turn anything 
# that used the output perceptron into a loop.

import math
import random
import time
import sys

rnd = random.Random(int(round(time.time() * 1000)))

 # dot product
 # uses smaller length if vector lengths are different
def dot(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))

 # sigmoid function
 # returns a value between 0 and 1. -> 0 as input -> -infinity and -> 1 as input -> infinity 
def sigmoid(x):
    if x <  -10: # cutoff to reduce calculations and prevent errors
        return 0
    if x > 10:
            return 1
    return 1.0 / (1.0+math.e**(-x))

 # generalized perceptron class that uses the sigmoid function
class perceptron:
    def __init__(self, length = 2):
        self.weights = []
        for i in range(length+1):
            self.weights.append(rnd.random() -.5)
    def out(self, input_vector):
        #print str(input_vector) + ", " + str(self.weights)
        return sigmoid(dot(input_vector, self.weights))
        #return dot(input_vector, self.weights)

 # generalized neural net
class net:
     # specify number of first layer inputs, number of hidden perceptrons, and learning rate in declaration 
    def __init__(self, inputs, hiddenPerceptrons, outputPerceptrons, learning_rate): 
        self.hiddenLayer = []
        for i in range(hiddenPerceptrons):
            self.hiddenLayer.append(perceptron(inputs))
        self.outputLayer = []
        for i in range(outputPerceptrons):
            self.outputLayer.append(perceptron(hiddenPerceptrons))
        self.learning_rate = learning_rate
    def out(self, input_vector): # this gives the output of the net without changing any weights
        hiddenOutputs = [1]
        for p in self.hiddenLayer:
            hiddenOutputs.append(p.out(input_vector))
        outputs = []
        for p in self.outputLayer:
            outputs.append(p.out(hiddenOutputs))
        return [hiddenOutputs, outputs]
    def train(self, input_vector, desired_outputs): # this will train the neural net on a single input_vector, iterating all of the values once
        hiddenOutputs, results = self.out(input_vector)
        errors = [(desired_output - result) for desired_output, result in zip(desired_outputs, results)]
        n = 0
        for error, result in zip(errors, results):
            delta = self.learning_rate * error * result * (1.0-result)
            for i, p in enumerate(self.hiddenLayer):
                hiddenDelta = delta * self.outputLayer[n].weights[i+1] * hiddenOutputs[i+1] * (1-hiddenOutputs[i+1])
                for i2, value in enumerate(input_vector):
                    p.weights[i2] += float(hiddenDelta)*float(value)
            for i, value in enumerate(hiddenOutputs):
                self.outputLayer[n].weights[i] += delta * value
            n+=1
        return errors # returns the error on this input
    def randomize(self): # reset all of the weights to random numbers
        for p in self.outputLayer:
            for i, w in enumerate(p.weights):
                p.weights[i] = (rnd.random() -.5)  
        for p in self.hiddenLayer:
            for i, w in enumerate(p.weights):
                p.weights[i] = (rnd.random() -.5)  
    def toString(self): # this makes it easy to print the final net
        s = "Hidden Layer:\n"
        for p in self.hiddenLayer:
            s = s + str(p.weights) + "\n"
        s = s + "Output Layer:\n"
        for p in self.outputLayer:
            s = s + str(p.weights) + "\n"
        return s

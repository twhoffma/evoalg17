'''
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
'''
import numpy as np
global bias

#Will contain weights for all perceptrons
global weights 
global inputs_bias

class mlp:
    def __init__(self, inputs, targets, nhidden):
        self.beta = 1
        self.eta = 0.1
        self.momentum = 0.0
        #inputs are the data rows
        #inputs[0] is one sensory observation.
        global bias 
        bias = -np.ones((np.shape(inputs)[0],1))
        global inputs_bias 
        inputs_bias = np.concatenate((inputs, bias), axis=1)
        global weights 
        weights = []
        n = np.shape(inputs_bias)[1]
        #Initialize input and hidden layer(s)' weights
        for i in range(1,3):
            weights.append(np.random.uniform(-1/np.sqrt(n), 1/np.sqrt(n), (n, nhidden)))
        print(np.shape(inputs_bias))
        print(np.shape(weights))
        print(weights[0])
        print('"init": To be implemented')

    # You should add your own methods as well!

    def earlystopping(self, inputs, targets, valid, validtargets):
        print('"earlystopping": To be implemented')

    def train(self, inputs, targets, iterations=100):
        print('"train": To be implemented')
        #We need to implement shuffling at each iteration
        #All need to be iterated over
        print(np.shape(inputs))

    def forward(self, inputs):
        print('"forward": To be implemented')
        outputs = inputs
        #activation function
        fn = np.vectorize(lambda t: return(1 if t > 0 else 0))
        for weight in weights:
            outputs = np.dot(outputs, weights)
            
        print(np.shape(output))

    def confusion(self, inputs, targets):
        print('"confusion": To be implemented')
    
    def calcError(y, t):
        print("Determine the error")
    
    def sigmoid(x, beta=3):
        return(1/(1+np.exp(-beta * x)))
    
    def delta0(w, t, beta=3):
        print("First derivative of sigmoid function")
    
    def deltah(w, t, beta=3):
        print("Second derivative of sigmoid function")


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
        #global bias 
        #bias = -np.ones((np.shape(inputs)[0],1))
        #global inputs_bias 
        #inputs_bias = np.concatenate((inputs, bias), axis=1)
        inputs_bias = self.add_bias(inputs)
        global weights 
        weights = []
        n = np.shape(inputs_bias)[1]
        #Initialize input and hidden layer(s)' weights
        for i in range(1,3):
            weights.append(np.random.uniform(-1/np.sqrt(n), 1/np.sqrt(n), (n, nhidden)))
        #weights must be initialized differently for output layer, hidden layer. 
        #Hidden layer default = 12, while output defaults = 8. 
        #print(np.shape(inputs_bias))
        #print(np.shape(weights))
        #print(weights[0])
        #print('"init": To be implemented')

    # You should add your own methods as well!
    def add_bias(self, inputs):
        bias = -np.ones((np.shape(inputs)[0],1))
        return(np.concatenate((inputs, bias), axis=1))
     
    def earlystopping(self, inputs, targets, valid, validtargets):
        print('"earlystopping": To be implemented')
        #self.forward(inputs)

    def train(self, inputs, targets, iterations=100):
        print('"train": To be implemented')
        #We need to implement shuffling at each iteration
        #All need to be iterated over
        print(np.shape(inputs))
    
    def forward(self, inputs):
        print('"forward": To be implemented')
        outputs = self.add_bias(inputs)
        print(np.shape(outputs))
        print(np.shape(weights))
        #activation function
        #fn = np.vectorize(lambda t: return(1 if t > 0 else 0))
        fn = np.vectorize(lambda x: sigmoid(x)) 
        print(np.shape(outputs))
        for weight in weights:
            outputs = np.dot(outputs, weight)
            #First time, the output is vectors one for each of the neurons in the perceptron. Default 12.
            #The next time the it is 227 times 12 and multiply with 12 times 8 to get the correct outputs. 
            print(outputs)

    def confusion(self, inputs, targets):
        #print(inputs)
        #print(targets)
        #TODO: Make a 8 x 8 matrix. Colums are targets, Rows are selected. Add percent like a correlation table.
        print('"confusion": To be implemented')
    
    def calcError(y, t):
        print("Determine the error")
    
    def sigmoid(x):
        return(1/(1+np.exp(-self.beta * x)))
    
    def delta0(w, t):
        #Remember self.beta
        print("First derivative of sigmoid function")
    
    def deltah(w, t):
        print("Second derivative of sigmoid function")
        #Remember self.beta


'''
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
'''
import numpy as np

class mlp:
    def __init__(self, inputs, targets, nhidden, momentum = 0, eta = 0.1, beta = 1):
        self.beta = beta
        self.eta = eta
        self.momentum = momentum
        self.weights = []
        #self.bias = []
        #self.inputs_bias = self.add_bias(inputs)
        self.last_presignals = []
        self.act_fn = np.vectorize(lambda x: self.sigmoid(x))
        self.deriv_act_fn = np.vectorize(lambda x: self.deriv_sigmoid(x))
         
        n = np.shape(inputs)[1] + 1
        m = np.shape(targets)[1]
         
        #Hidden layer initial weights, including bias
        self.weights.append(np.random.uniform(-1/np.sqrt(n), 1/np.sqrt(n), (n, nhidden)))
        
        #Output layer initial weights, including bias
        self.weights.append(np.random.uniform(-1/np.sqrt(m), 1/np.sqrt(m), (nhidden + 1, m)))
    
    # You should add your own methods as well!
    def add_bias(self, inputs):
        a = len(np.shape(inputs))-1
        if a == 0:
            bias = [-1]
        else:
            bias = -np.ones((np.shape(inputs)[0],1))
        #print("Axis=" + str(a))
        #print(np.shape(inputs))
        #print(np.shape(bias))
        out = np.concatenate((inputs, bias), axis=a)
        #print(out)
        return(out)
     
    def earlystopping(self, inputs, targets, valid, validtargets, iterations=1):
        print('"earlystopping": To be implemented')
        #So this is the triggering for training
        self.train(inputs[0:1], targets[0], iterations)
        

    def train(self, inputs, targets, iterations=100):
        print('Training')
        #We need to implement shuffling at each iteration
        #All need to be iterated over
        #print(np.shape(inputs))
        
        prev_weight_chgs = []
        for weight in self.weights: 
            prev_weight_chgs.append(np.zeros(np.shape(weight)))
        
        for iter in range(iterations):
            if len(np.shape(inputs)) <= 1:
                inputs = [inputs] #Only used for testing
                targets = [targets]
            
            #Randomize the order to improve training.
            training_order = list(range(0, np.shape(inputs)[0]))
            np.random.shuffle(training_order)
            
            #We train and adjust weights a single sample at a time.
            for o in training_order:
                i = inputs[o]
                t = targets[o]
                a = self.forward(i)
                #err =
                print(self.calc_error(a,t)) 
                self.calc_output_weights_chg(a,t)
                #First we do the output layer, which is the last presignal
                #delta_k = self.delta_output(a,t)
                #print("<delta_k>")
                #print(delta_k)
                #print("</delta_k>")
                #delta_k is a [1...8] array
                #if k is the position, we need to update 12 values for each k. 
                #The 12 inbound weights to this output node.
                
                #presignal = self.last_presignals[-1]
                #chg = prev_weight_chgs[-1]
                
                #if len(np.shape(inputs)) <= 1:
                #    presignal = [presignal]
                #
                #print("chg")
                #print(np.shape(chg))
                #print("presig")
                #print(np.shape(presignal))
                #print("delta_k")
                #print(np.shape(delta_k))
                #for k in range(0, len(delta_k)):
                #    for j in range(0, len(presignal)):
                #        chg[j][k] = self.act_fn(presignal[j][k])*delta_k[k]
                #print(chg)
                #        
                #print("Presignals")
                #print(presignal)
                #print(self.last_presignals)
                #print("Activation values") 
                #print(a)
                
    def calc_output_weights_chg(self, a, t):
       ps = self.last_presignals[-1]
       preHidden = self.last_presignals[0] #We have only one hidden layer
       actHidden = self.act_fn(preHidden)
       deltas = self.delta_output(a, t)
       print(np.shape(a))
       print(np.shape(actHidden))
     
    def forward(self, inputs):
        outputs = inputs
        
        #Keep the last presignals for each last
        self.last_presignals = []
        for weight in self.weights:
            outputs = self.add_bias(outputs)
            outputs = np.dot(outputs, weight)
            print("out")
            print(outputs) 
            self.last_presignals.append(outputs)
            
            outputs = self.act_fn(outputs)
        return outputs
    
    def confusion(self, inputs, targets):
        #print(inputs)
        #print(targets)
        #TODO: Make a 8 x 8 matrix. Colums are targets, Rows are selected. Add percent like a correlation table.
        print('"confusion": To be implemented')
    
    def calc_error(self, y, t):
        e = 0.5*sum(np.square(np.subtract(y,t)))
        return(e)
    
    def sigmoid(self, x):
        return(1/(1+np.exp(-self.beta * x)))
    
    def deriv_sigmoid(self, x):
        return(self.sigmoid(x)*(1-self.sigmoid(x)))
    
    def delta_output(self, a, t):
        z_ks = self.last_presignals[-1]
        diffs = np.subtract(a,t)
        acts = self.deriv_act_fn(z_ks)
        deltas = np.multiply(diffs, acts)
        return(deltas)
     
    def delta_hidden(self, w, t):
        print("Second derivative of sigmoid function")
        #Remember self.beta


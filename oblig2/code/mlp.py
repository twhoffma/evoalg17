'''
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
'''
import numpy as np

class mlp:
    def __init__(self, inputs, targets, nhidden, momentum = 0, eta = 0.1, beta = 1, output_latex = False):
        self.beta = beta
        self.eta = eta
        self.momentum = momentum
        self.weights = []
        #self.bias = []
        #self.inputs_bias = self.add_bias(inputs)
        self.last_presignals = []
        self.act_fn = np.vectorize(lambda x: self.sigmoid(x))
        self.deriv_act_fn = np.vectorize(lambda x: self.deriv_sigmoid(x))
        
        #Output options
        self.output_latex = output_latex 
        
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
        out = np.concatenate((inputs, bias), axis=a)
        return(out)
     
    def earlystopping(self, inputs, targets, valid, validtargets, iterations=100):
        #self.train2(inputs, targets, iterations)
        it = 0
        avg_errors = []
        while it < iterations and self.errors_converge(avg_errors):
            print(it)
            self.train(inputs, targets, 10)
            validres = self.forward(valid)
            err = []
            for i in range(np.shape(validres)[0]):
                err.append(self.calc_error(validres[i], validtargets[i]))
            avg_errors.append(np.average(err))
            it = it + 1
            #print(avg_errors[-1])
        print("Stopped after {} iterations".format(it))
    
    #Helper function to check if errors converge  
    def errors_converge(self, avg_errors):
        #We measure backwards
        if len(avg_errors) < 3:
            return(True)
        #If there is improvement for at least one the last two epochs, keep going.
        elif ((avg_errors[-2] - avg_errors[-1] > 0.001) or (avg_errors[-3] - avg_errors[-2] > 0.001)):
            return(True)
        else:
            #Stop if no improvement
            return(False)
        
    def train2(self, inputs, targets, iterations=100):
        #print('Training')
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
            
            errs = [] 
            #Randomize the order to improve training.
            training_order = list(range(0, np.shape(inputs)[0]))
            np.random.shuffle(training_order)
            #We train and adjust weights a single sample at a time.
            for o in training_order:
                i = inputs[o]
                t = targets[o]
                a = self.forward(i)
                #err =
                errs.append(self.calc_error(a,t))
                wchg_o = self.calc_output_weights_chg(a,t)
                wchg_h = self.calc_hidden_weights_chg(a,t, i)
                #First we do the output layer, which is the last presignal
                #delta_k = self.delta_output(a,t)
                #print("<delta_k>")
                #print(delta_k)
                #print("</delta_k>")
                #delta_k is a [1...8] array
                #if k is the position, we need to update 12 values for each k. 
                #The 12 inbound weights to this output node.
                self.weights[-1] = np.subtract(self.weights[-1],  np.multiply(self.eta, wchg_o))
                self.weights[-2] = np.subtract(self.weights[-2],  np.multiply(self.eta, wchg_h))
                #print(np.shape(self.weights[-2]))
                #print(np.shape(self.weights[-1]))
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
            print(np.average(errs))
    
    def train(self, inputs, targets, iterations):
        prev_weight_changes = []
        for weight in self.weights: 
            prev_weight_changes.append(np.zeros(np.shape(weight)))
        
        for it in range(iterations):
            errs = [] 
            #Randomize the order to improve training.
            training_order = list(range(0, np.shape(inputs)[0]))
            np.random.shuffle(training_order)
            
            #We train and adjust weights a single sample at a time.
            for o in training_order:
                i = inputs[o]
                t = targets[o]
                a = self.forward(i)
                errs.append(self.calc_error(a,t))
                weights_change_output = self.calc_output_weights_chg(a,t)
                weights_change_hidden = self.calc_hidden_weights_chg(a,t, i)
                
                weights_change_output = np.multiply(self.eta, weights_change_output)
                weights_change_hidden = np.multiply(self.eta, weights_change_hidden)
                 
                #self.weights[-1] = np.subtract(self.weights[-1],  np.multiply(self.eta, wchg_o))
                #self.weights[-2] = np.subtract(self.weights[-2],  np.multiply(self.eta, wchg_h))
                self.weights[-1] = np.subtract(self.weights[-1],  weights_change_output)
                self.weights[-2] = np.subtract(self.weights[-2],  weights_change_hidden)
                
                #Add momentum from previous runs
                self.weights[-1] = np.add(self.weights[-1], np.multiply(prev_weight_changes[-1], self.momentum))
                self.weights[-2] = np.add(self.weights[-2], np.multiply(prev_weight_changes[-2], self.momentum))
                prev_weight_changes[-1] = weights_change_output
                prev_weight_changes[-2] = weights_change_hidden
            #print(np.average(errs))
    
    def calc_output_weights_chg(self, a, t):
        ps = self.last_presignals[-1]
        preHidden = self.last_presignals[-2] #We have only one hidden layer
        actHidden = self.act_fn(preHidden)
        deltas = self.delta_output(a, t)
        #print(np.shape(a))
        #print(np.shape(actHidden))
        wght_chg = np.zeros((len(actHidden) + 1,len(deltas))) 
        #Output weights' change
        for i in range(0,len(actHidden)):
            for j in range(0, len(deltas)):
                wght_chg[i][j] = deltas[j]*actHidden[i]
        #Weight update of betas' weights
        for j in range(0, len(deltas)):
            wght_chg[len(actHidden)][j] = deltas[j]
        #print(wght_chg)
        return(wght_chg)
    
    def calc_hidden_weights_chg(self, a, t, input_data):
        d_o = self.delta_output(a, t)
        w_o = np.array(self.weights[-1]) 
        #deltas = self.calc_hidden_deltas(
        #wght_chg = np.zeros((len(input_data) + 1,len(deltas))) #one additional row for bias
        d_h = self.delta_hidden(a,t)
        hidden_weight_changes = np.zeros((len(input_data)+1,len(d_h)))
        for i in range(0, len(input_data)):
            for j in range(0, len(d_h)):
                hidden_weight_changes[i][j] = input_data[i]*d_h[j]
        for j in range(0, len(d_h)):
            hidden_weight_changes[len(input_data)][j] = d_h[j]
        #print(hidden_weight_changes)
        return(hidden_weight_changes)
        #print(w_o)
        #print(w_o[:,0])
 
    def forward(self, inputs):
        outputs = inputs
        
        #Keep the last presignals for each last
        self.last_presignals = []
        for weight in self.weights:
            outputs = self.add_bias(outputs)
            outputs = np.dot(outputs, weight)
            #print("out")
            #print(outputs) 
            self.last_presignals.append(outputs)
            
            outputs = self.act_fn(outputs)
        return outputs
    
    def confusion(self, inputs, targets):
        print('Confusion matrix:')
        cats = self.forward(inputs)
        confusion_table = np.zeros((8,8))
        num_correct = 0
        num_total = 0
        for i in range(0, np.shape(cats)[0]):
            x = cats[i].argmax()
            y = targets[i].argmax()
            if x == y:
                num_correct = num_correct + 1
            confusion_table[y][x] = confusion_table[y][x] + 1
        for i in range(0, np.shape(confusion_table)[0]):
            confusion_table[i] = np.round(np.divide(confusion_table[i], sum(confusion_table[i])),2)
        if self.output_latex:
            self.print_latex_table(confusion_table)
        else:
            print(confusion_table)
        print("Success percentage: {:.2f}%".format(float(num_correct) / np.shape(targets)[0] * 100))
    
    def print_latex_table(self, tbl):
        cols = range(np.shape(tbl)[1])
        print("\\begin{tabular}{|l"+str.join("", ["|c" for x in cols])+"|}")
        print("target/out & " + str.join(" & ", [str(x) for x in cols]) + " \\\\") 
        print("\\hline")
        for i in range(np.shape(tbl)[0]):
            r = tbl[i]
            formatted = ["{:.2f}".format(x) for x in r]
            print(str(i) + " & " + str.join(" & ", ["\\textbf{" + x + "}" if x != "0.00" else x for x in formatted]) + "\\\\")
            print("\\hline")
        print("\\hline")
        print("\\end{tabular}")
    
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
     
    def delta_hidden(self, a, t):
        weights_o = self.weights[-1] #Output layer weights
        delta_o = self.delta_output(a,t)
        z_hidden = self.last_presignals[-2]
        a_hidden = self.deriv_act_fn(z_hidden)
        d_out_weighted = np.zeros(len(z_hidden))
        #Since z_hidden is one shorter than the weights, delta hidden will be nHidden long
        for i in range(0, len(z_hidden)):
            d_out_weighted[i] = np.dot(delta_o, weights_o[i])
        d_hidden = np.multiply(a_hidden, d_out_weighted)
        #print(d_hidden)
        return(d_hidden)
        #print("delta hidden size:")
        #print(np.shape(delta_hidden))
        #print("--del--")
        #print(delta_o_weighted)
        #print(np.shape(weights_o))
        #print(np.shape(delta_o)) 
        #print(np.shape(z_hidden))
        #print(np.shape(a_hidden))

import numpy as np
import math

class LSTMWeights:

    def __init__ (self, N =1):
        # weight initialization
        x = math.sqrt(2/N)
        # weights
        self.w = np.random.uniform(-x, x, (N, 2, 4))
        #biases
        self.b = np.zeros((N,4))
        self.b[:,0] = 1.0
        
        #output layer weights
        self.Wy = np.random.uniform(-x, x, N) 
        self.by = 0
        
        
        self.step = 0.0001
        self.cache = {}
        self.N = N


    def set_step(self, x):
        self.step = x

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) 

    def tanh(self,x):
        return np.tanh(np.clip(x, -500, 500))
        

    #calculates the loss that the model currently gets with MSE
    def simulate(self, data, shouldMse = False):
        
        #duplicate inputs for vector multiplication
        X = np.array([data]*self.N).T
        
        # states for each timestep
        H = np.zeros((len(data)+1,self.N))
        C = np.zeros((len(data)+1,self.N))
        
        # pre-activation function values, good for vector mult & cached for reuse in bptt
        A = np.zeros((len(data),self.N,4))
        
        f = np.zeros((len(data), self.N))
        inp = np.zeros((len(data), self.N))
        c = np.zeros((len(data), self.N))
        o = np.zeros((len(data), self.N))

        # outputs
        Y = np.zeros(len(data))
        
        #mean squared error
        MSE = 0.0
        
        #forward pass
        for i in range(len(data)):
            A[i] = H[i][:,None] * self.w[:,0,:] + X[i][:,None] * self.w[:,1,:] + self.b
            f[i] = self.sigmoid(A[i,:,0])
            inp[i] = self.sigmoid(A[i,:,1])
            c[i] = self.tanh(A[i,:,2])
            o[i] = self.sigmoid(A[i,:,3])
            C[i+1] = f[i] * C[i] + inp[i] * c[i]
            H[i+1] = o[i] * self.tanh(C[i+1])
            Y[i] = H[i + 1] @ self.Wy + self.by
        self.cache = (X, H, C, A, f, inp, c, o, Y)
        
        #calculate Mse only if necesarry    
        if shouldMse:
            MSE = np.sum((Y[:len(data)-1] - data[1:])**2) / (len(data)-1)
                            
        return MSE


    #back propagation through time
    def bptt(self, data, steps, step_size = None):
        if step_size is not None:
            self.step = step_size
        for i in range(steps):
            dh = np.zeros((len(data)+1, self.N))
            dc = np.zeros((len(data)+1, self.N))
            wgrad = np.zeros((self.N, 2, 4))
            bgrad = np.zeros((self.N, 4))
            self.simulate(data)
            X,H,C,A,f,inp,c,o,Y = self.cache
            # dh[1:len(data)] = 2 * (H[1:len(data)] - X[1:])
            
            Wygrad = np.zeros(self.N)
            bygrad = 0.0
            
            dY = 2 * (Y[:len(data)-1] - data[1:])   # shape: (T-1,)

            for t in range(len(data)-1):
                Wygrad += H[t+1] * dY[t]
                bygrad += dY[t]
                dh[t+1] += dY[t] * self.Wy
            
            for j in range(len(data)-2,-1,-1):
                dc[j+1] += dh[j+1]*o[j] * (1-self.tanh(C[j+1])**2)
                
                da_f = f[j] * (1 - f[j])
                da_i = inp[j] * (1 - inp[j])
                da_c = 1 - c[j]**2
                da_o = o[j] * (1 - o[j])
                
                wpaths = np.zeros((self.N,4))
                wpaths[:,0] = dc[j+1] * C[j] * da_f
                wpaths[:,1] = dc[j+1] * c[j] * da_i
                wpaths[:,2] = dc[j+1] * inp[j] * da_c
                wpaths[:,3] = dh[j+1] * self.tanh(C[j+1]) * da_o
                
                wgrad[:,0] += wpaths * H[j][:,None]
                wgrad[:,1] += wpaths * X[j][:,None]
                bgrad += wpaths 
                
                if j>0:
                    dc[j] += dc[j+1] * f[j]
                    dh[j] += np.sum(wpaths*self.w[:,0,:], axis = 1)
            
            #clip grad
            max_grad_norm = 1.0
            grad_norm = np.sqrt(np.sum(wgrad**2) + np.sum(bgrad**2) + np.sum(Wygrad**2) + bygrad**2)
            if grad_norm > max_grad_norm:
                scale = max_grad_norm / grad_norm
                wgrad *= scale
                bgrad *= scale
                Wygrad *= scale
                bygrad *= scale
            self.w -= wgrad * self.step 
            self.b -= bgrad * self.step  
            self.Wy -= self.step * Wygrad
            self.by -= self.step * bygrad            
                
        return self.simulate(data, True)
            
                
                


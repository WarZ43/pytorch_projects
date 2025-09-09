import numpy
import math

class RNNWeights :
    def __init__(self, N = 1):
        self.N = N
        #xavier
        x = math.sqrt((6)/(2+2))
        self.w = numpy.random.rand(N,2)*2*x - x
        self.v = numpy.random.rand(N)*2*x -x
        self.b = numpy.zeros(N)
        self.c = 0
        self.cache = {}
        self.step_size = 0.0001
        
    def simulate(self, data, shouldMse = False):
        H = numpy.zeros((len(data)+1, self.N))
        Y = numpy.zeros((len(data)))
        for i in range(len(data)):
            H[i+1] = numpy.tanh(self.w[:,0] * H[i] + self.w[:,1] * data[i] + self.b)
            Y[i] = numpy.sum(self.v * H[i+1]) + self.c
        self.cache = (H , Y)
        
        MSE = 0
        if shouldMse:
            MSE = numpy.sum((Y[:len(data)-1] - data[1:])**2)
                    
        MSE /= len(data)
        
        return MSE
    
    def bptt(self, data, steps, step_size = 0.0001):
        self.step_size = step_size
        for i in range(steps):
            self.simulate(data)
            H, Y = self.cache
            wgrad = numpy.zeros((self.N, 2))
            vgrad = numpy.zeros(self.N)
            bgrad = numpy.zeros(self.N)
            cgrad = numpy.zeros(self.N)
            dY = 2*(Y[:len(data)-1]-data[1:]) # T-1 x 
            vgrad = numpy.sum(H[1:len(data-1)] * dY[:,None], axis = 0)
            cgrad = numpy.sum(dY)
            dH = dY[:,None] @ self.v[:,None].T # T-1 x N
            for j in range(len(data)-2, 0, -1):
                dA = dH[j]*(1-H[j+1]**2)
                wgrad[:,0] += dA * H[j]
                wgrad[:,1] += dA * data[j]
                bgrad += dA
            self.w -= wgrad * self.step_size / self.N 
            self.v -= vgrad * self.step_size / self.N
            self.b -= bgrad * self.step_size / self.N
            self.c -= cgrad * self.step_size / self.N
        return self.simulate(data,True)
            
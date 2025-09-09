import numpy as np
import math
import matplotlib.pyplot as plt

# Paste the entire corrected LSTMWeights class here...
class LSTMWeights:
    # ... (code from above) ...
    def __init__ (self, N =1):
        # Xavier initialization
        x = math.sqrt(1/N)
        # weights
        self.w = np.random.rand(N,2,4) * 2 * x - x
        #biases
        self.b = np.zeros((N,4))
        
        #output layer weights
        self.Wy = np.random.randn(N) * 0.01
        self.by = 0.0
        
        self.step = 0.0001
        self.cache = {}
        self.N = N

    def set_step(self, x):
        self.step = x

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def tanh(self,x):
        return np.tanh(x)
        
    def simulate(self, data, shouldMse = False):
        X = np.array([data]*self.N).T
        H = np.zeros((len(data)+1,self.N))
        C = np.zeros((len(data)+1,self.N))
        A = np.zeros((len(data),self.N,4))
        f = np.zeros((len(data), self.N))
        inp = np.zeros((len(data), self.N))
        c = np.zeros((len(data), self.N))
        o = np.zeros((len(data), self.N))
        Y = np.zeros(len(data))
        MSE = 0.0
        for i in range(len(data)):
            A[i] = H[i][:,None] * self.w[:,0,:] + X[i][:,None] * self.w[:,1,:] + self.b
            f[i] = self.sigmoid(A[i,:,0])
            inp[i] = self.sigmoid(A[i,:,1])
            c[i] = self.tanh(A[i,:,2])
            o[i] = self.sigmoid(A[i,:,3])
            C[i+1] = f[i] * C[i] + inp[i] * c[i]
            H[i+1] = o[i] * np.tanh(C[i+1])
            Y[i] = H[i + 1] @ self.Wy + self.by
        self.cache = (X, H, C, A, f, inp, c, o, Y)
        if shouldMse and len(data) > 1:
            MSE = np.sum((Y[:len(data)-1] - data[1:])**2)
            MSE /= (len(data) - 1)
        return MSE

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
            Wygrad = np.zeros(self.N)
            bygrad = 0.0
            if len(data) < 2:
                return 0.0
            n_terms = len(data) - 1
            dY = (Y[:len(data)-1] - data[1:]) / n_terms
            for t in range(n_terms):
                Wygrad += H[t+1] * dY[t]
                bygrad += dY[t]
                dh[t+1] += dY[t] * self.Wy
            for j in range(len(data)-2, -1, -1):
                dc[j+1] += dh[j+1]*o[j] * (1-np.tanh(C[j+1])**2)
                da_f = f[j] * (1 - f[j])
                da_i = inp[j] * (1 - inp[j])
                da_c = 1 - c[j]**2
                da_o = o[j] * (1 - o[j])
                wpaths = np.zeros((self.N,4))
                wpaths[:,0] = dc[j+1] * C[j] * da_f
                wpaths[:,1] = dc[j+1] * c[j] * da_i
                wpaths[:,2] = dc[j+1] * inp[j] * da_c
                wpaths[:,3] = dh[j+1] * np.tanh(C[j+1]) * da_o
                wgrad[:,0] += wpaths * H[j][:,None]
                wgrad[:,1] += wpaths * X[j][:,None]
                bgrad += wpaths 
                if j>0:
                    dc[j] += dc[j+1] * f[j]
                    dh[j] += np.sum(wpaths*self.w[:,0,:], axis = 1)
            self.w -= wgrad * self.step 
            self.b -= bgrad * self.step  
            self.Wy -= self.step * Wygrad
            self.by -= self.step * bygrad            
        return self.simulate(data, True)

# --- Main training script ---
# Create an "easy" training set: a simple sine wave
train_data = np.array( [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
], dtype = float)

train_data /= 622

# Instantiate the model with 8 hidden units
# Using more than 1 unit often helps convergence
lstm = LSTMWeights(N=8)

# Train the model
epochs = 200
learning_rate = 0.5 # A higher learning rate works well with normalized gradients
losses = []

print("Starting training...")
for epoch in range(epochs):
    # We can train in batches of steps
    loss = lstm.bptt(train_data, steps=10, step_size=learning_rate)
    losses.append(loss)
    if (epoch) % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.8f}")

print(f"Final Loss: {losses[-1]:.8f}")

# Plot the losses
plt.plot(losses)
plt.xlabel("Epochs (x10 steps)")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Time")
plt.show()
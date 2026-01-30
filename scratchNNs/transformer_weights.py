import numpy as np
import math

class TransformerWeights:
    """
    From scratch Transformer implementation for time-series numerical prediction using SGD, with a 2 layer ReLU FFN and residual connections. 
    Hyperparameters include:
    - n_features: Number of input features per time step
    - d_model: Dimension of the model
    - n_heads: Number of attention heads
    - seq_len: Length of the input sequences
    """
    def __init__(self, n_features=1, d_model=16, n_heads=4):
        self.n_features = n_features 
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.step = 0.0001
        self.cache = {}

        # initialize weights xavier style
        scale = math.sqrt(2.0 / d_model)

        # projection weights
        self.W_proj = np.random.uniform(-scale, scale, (n_features, d_model))
        self.b_proj = np.zeros(d_model)

        # attention weights
        self.W_q = np.random.uniform(-scale, scale, (d_model, d_model))
        self.W_k = np.random.uniform(-scale, scale, (d_model, d_model))
        self.W_v = np.random.uniform(-scale, scale, (d_model, d_model))
        
        # output projection weights
        self.W_o = np.random.uniform(-scale, scale, (d_model, d_model))
        
        # feedforward network parameters
        self.d_ff = d_model * 4  # Expansion dimension

        # FFN Weights
        self.W_ff1 = np.random.uniform(-scale, scale, (d_model, self.d_ff)) # Expand
        self.b_ff1 = np.zeros(self.d_ff)
        self.W_ff2 = np.random.uniform(-scale, scale, (self.d_ff, d_model)) # Contract
        self.b_ff2 = np.zeros(d_model)
        
        #final fc weights
        self.W_final =  np.random.uniform(-scale, scale, (d_model, 1))
        self.b_final = np.zeros(1)

    def set_step(self, x):
        self.step = x

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    def dsoftmax(self, d_output, softmax_output):
        sum_term = np.sum(d_output * softmax_output, axis=-1, keepdims=True)
        return softmax_output * (d_output - sum_term)
    def relu(self, x):
        return np.maximum(0, x)

    def get_pos_encoding(self, length):
        """Generates sinusoidal position encodings"""
        pe = np.zeros((length, self.d_model))
        position = np.arange(length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe

    def simulate(self, data, targets):
        """
        Forward pass with Batch Support.
        data: (Batch_Size, Seq_Len, N_Features) OR (Seq_Len, N_Features)
        targets: (Batch_Size, Seq_Len) OR (Seq_Len,)
        """
        # ensure data and targets have batch dimension
        X_in = np.array(data)
        if X_in.ndim == 1:
            X_in = X_in[np.newaxis, :, np.newaxis]
        elif X_in.ndim == 2:
            X_in = X_in[np.newaxis, :, :]
        targets = np.array(targets)
        if targets.ndim == 1:
            targets = targets[np.newaxis, :]
            
        #record input dimensions
        B, S, F = X_in.shape
        
        # feature projection & pos encoding
        # (B, S, F) @ (F, D) -> (B, S, D)
        X = X_in @ self.W_proj + self.b_proj
        pos_enc = self.get_pos_encoding(S)
        X += pos_enc

        # compute Q, K, V matrices
        # (B, S, D) @ (D, D) -> (B, S, D)
        Q_full = X @ self.W_q
        K_full = X @ self.W_k
        V_full = X @ self.W_v

        # split into heads
        # reshape to (B, S, H, D_h) then transpose to (B, H, S, D_h)
        Q = Q_full.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K_full.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V_full.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)


        # calculate attention scores
        # (B, H, S, D_h) @ (B, H, D_h, S) -> (B, H, S, S)
        scores = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head)
        
        # create upper triangle mask to prevent access to future positions
        mask = np.triu(np.ones((S, S)), k=1) * -1e9
        scores += mask

        # softmax & aggregation
        attn_weights = self.softmax(scores)
        context_heads = attn_weights @ V  # (B, H, S, D_h)
        
        

        # concat heads
        # (B, H, S, D_h) -> (B, S, H, D_h) -> (B, S, D)
        context = context_heads.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)

        # output projection (mix heads)
        # (B, S, D) @ (D, D) -> (B, S, D)
        attn_out = context @ self.W_o
        H1 = attn_out + X  # residual connection
        ff_hidden = H1 @ self.W_ff1 + self.b_ff1  # (B, S, D_ff)
        ff_activated = self.relu(ff_hidden)  # (B, S, D_ff)
        ff_out = ff_activated @ self.W_ff2 + self.b_ff2  # (B, S, D)
        
        Z = ff_out + H1  # residual connection
        # (B, S, D) @ (D, 1) -> (B, S, 1)
        Y_out = Z @ self.W_final + self.b_final  # (B, S, 1)
        Y_out = Y_out.squeeze(-1)  # (B, S)

        MSE = np.mean((Y_out - targets) ** 2)
        
        # Cache for backpropagation
        self.cache = (X_in, X, Q_full, K_full, V_full, Q, K, V, scores, attn_weights, 
                      context_heads, context, attn_out, H1, ff_hidden, ff_activated, ff_out, Z, Y_out, MSE)

        return MSE
    def bp(self, data, targets, steps, step_size=0.0001):
        """
        Backpropagation through time for Transformer.
        data: (Batch_Size, Seq_Len, N_Features) OR (Seq_Len, N_Features)
        targets: (Batch_Size, Seq_Len) OR (Seq_Len,)
        """
        self.step = step_size
        for _ in range(steps):
            self.simulate(data, targets)
            (X_in, X, Q_full, K_full, V_full, Q, K, V, scores, attn_weights, 
             context_heads, context, attn_out, H1, ff_hidden, ff_activated, ff_out, Z, Y_out, MSE) = self.cache
            B, S = Y_out.shape
            #MSE -> Y
            # Y -> bf, Wf, Z  -- Crucial
            dY = 2 * (Y_out - targets) / Y_out.size  # (B, S) 
            dY = dY[:, :, np.newaxis]
            # Z -> Wo, context -- Crucial
            dW_final = Z.reshape(-1, self.d_model).T @ dY.reshape(-1, 1)
            db_final = np.sum(dY, axis=(0, 1)).flatten()
            dZ = dY @ self.W_final.T  # (B, S, D)
            
            d_ff_out = dZ
            dH1 = dZ
            dW_ff2 = ff_activated.reshape(-1, self.d_ff).T @ d_ff_out.reshape(-1, self.d_model)
            db_ff2 = np.sum(d_ff_out, axis=(0, 1))
            d_ff_activated = d_ff_out @ self.W_ff2.T # (B, S, D_ff)
            d_ff_hidden = d_ff_activated * (ff_hidden > 0)

            dW_ff1 = H1.reshape(-1, self.d_model).T @ d_ff_hidden.reshape(-1, self.d_ff)
            db_ff1 = np.sum(d_ff_hidden, axis=(0, 1))
            dH1_from_ffn = d_ff_hidden @ self.W_ff1.T 
            
            dH1 += dH1_from_ffn 

            d_attn_out = dH1
            dX = dH1
            # context -> heads
            dW_o = context.reshape(-1, self.d_model).T @ d_attn_out.reshape(-1, self.d_model)
            dcontext = d_attn_out @ self.W_o.T  # (B, S, H, D_h)
            dcontext_heads = dcontext.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)  # (B, H, S, D_h)
            # heads -> V, atten_weights -- Crucial
            dV = attn_weights.transpose(0, 1, 3, 2) @ dcontext_heads
            datten_weights = dcontext_heads @ V.transpose(0, 1, 3, 2)  # (B, H, S, S)
            # atten_weights -> scores
            dscores = self.dsoftmax(datten_weights, attn_weights)  # (B, H, S, S)
            # scores -> Q, K -- Crucial
            dQ = dscores @ K / math.sqrt(self.d_head)
            dK = dscores.transpose(0, 1, 3, 2) @ Q / math.sqrt(self.d_head)
            # Q, K, V -> Q_full, K_full, V_full
            dQ_full = dQ.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
            dK_full = dK.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
            dV_full = dV.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
            # Q_full, K_full, V_full -> X , W_q, W_k, W_v -- Crucial
            dW_q = X.reshape(-1, self.d_model).T @ dQ_full.reshape(-1, self.d_model)
            dW_k = X.reshape(-1, self.d_model).T @ dK_full.reshape(-1, self.d_model)
            dW_v = X.reshape(-1, self.d_model).T @ dV_full.reshape(-1, self.d_model)    
            
            dX_attn = dQ_full @ self.W_q.T + dK_full @ self.W_k.T + dV_full @ self.W_v.T
            dX += dX_attn
            
            # X -> W_proj, b_proj -- Crucial
            dW_proj = X_in.reshape(-1, X_in.shape[-1]).T @ dX.reshape(-1, dX.shape[-1])
            db_proj = np.sum(dX, axis=(0, 1))
            # Update weights
            self.W_final -= self.step * dW_final
            self.b_final -= self.step * db_final
            self.W_ff2 -= self.step * dW_ff2
            self.b_ff2 -= self.step * db_ff2
            self.W_ff1 -= self.step * dW_ff1
            self.b_ff1 -= self.step * db_ff1
            self.W_o -= self.step * dW_o
            self.W_q -= self.step * dW_q
            self.W_k -= self.step * dW_k
            self.W_v -= self.step * dW_v
            self.W_proj -= self.step * dW_proj
            self.b_proj -= self.step * db_proj
        
        return self.simulate(data, targets)
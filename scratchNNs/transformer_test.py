import numpy as np
from transformer_weights import TransformerWeights 

model = TransformerWeights(n_features=1, d_model=64, n_heads=4)
airline_passengers = np.array([
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
], dtype=float)

airline_passengers = airline_passengers / 622.0

inputs = airline_passengers[:-1]
targets = airline_passengers[1:]

print(f"Training on {len(inputs)} time steps...")

# using a lower learning rate usually helps Transformers converge stably
final_loss = model.bp(inputs, targets, steps=50000, step_size=0.0005)

print("-" * 30)
print(f"Final Loss (MSE): {final_loss:.5f}")
print("-" * 30)

# visualize predictions
predictions = model.cache[-2]
print("Sample Predictions vs Actuals (Last 5 steps):")
# reshape predictions to 1D to compare with our 1D targets
preds_flat = predictions.flatten()
targs_flat = targets.flatten()

for i in range(1, 6):
    print(f"Step {len(preds_flat)-i}: Pred: {preds_flat[-i]:.4f} | Actual: {targs_flat[-i]:.4f}")

# print weights
# warning: These matrices are much larger than LSTM weights due to multiple attention heads and projections
shouldPrint = False
if shouldPrint:
    print("-" * 30)
    print("Weight Shapes:")
    print(f"W_q: {model.W_q.shape}")
    print(f"W_final: {model.W_final.shape}")
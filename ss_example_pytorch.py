import torch
# Define the system matrices
A = torch.tensor([[0.1]], dtype=torch.float32)
B = torch.tensor([[1.0]], dtype=torch.float32)
C = torch.tensor([[1.0]], dtype=torch.float32)
D = torch.tensor([[0.0]], dtype=torch.float32)  # Often zero in many systems
# Initial state
x = torch.tensor([[0.0]], dtype=torch.float32)

# Input (for example, step input)
u = torch.tensor([[1.0]], dtype=torch.float32)
# Simulate one time step
x_next = A @ x + B @ u
y = C @ x + D @ u

print("Next state:", x_next)
print("Output:", y)

import triton
import triton.language as tl
import torch
import math
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10,4)})
sns.set_style("whitegrid", {'axes.grid' : False})
# Utility functions for creating tensors on the GPU.
ones = lambda *size: torch.ones(*size).float().cuda()  # Creates a tensor of ones.
zeros = lambda *size: torch.zeros(*size).float().cuda()  # Creates a tensor of zeros.
arange = lambda n: torch.arange(n).float().cuda()  # Creates a range tensor from 0 to n-1.
rand = lambda size: torch.rand(*size).abs().float().cuda()  # Creates a tensor of random numbers.
x = torch.zeros(4)
y = torch.zeros(4)
def check(*inputs, prec=1e-4):
    """
    Checks if pairs of tensors are approximately equal up to a precision.
    
    Args:
    *inputs: A list of tensors to compare in pairs. Each pair is checked for closeness.
    prec: The precision tolerance for comparisons.
    
    Raises:
    AssertionError if any pair of tensors are not close up to the specified precision.
    
    Outputs:
    Prints a verification message if all pairs are close.
    """
    for i, (a, b) in enumerate(zip(inputs[::2], inputs[1::2])):
        if isinstance(b, list):
            b = torch.tensor(b)
        c = torch.allclose(a.cpu(), b.cpu(), prec)
        c1 = torch.isclose(a.cpu(), b.cpu(), prec)
        assert c, f"{i}\n{a}\n{b}\n{c1}"
    print("check")
tensor = check(x,y)
print(tensor)
@triton.jit
def triton_hello_world(X, Y, Z, K: tl.constexpr, L: tl.constexpr):
    """
    A Triton kernel that performs element-wise addition of two matrices X and Y, storing the result in Z.
    
    Args:
    X: A 1D tensor representing the first matrix to add.
    Y: A 2D tensor representing the second matrix to add.
    Z: A 2D tensor where the result of the addition will be stored.
    K: The size of the inner dimension for X and Y.
    L: The size of the outer dimension for Y and Z.
    
    This function exploits data parallelism by performing the addition in a vectorized manner.
    """
    # Use arange to build the shape for loading
    Ks = tl.arange(0, K) # K
    Ls = tl.arange(0, L)[:, None] # L x 1

    # Load from memory
    x = tl.load(X + Ks)  # Load elements of X
    y = tl.load(Y + Ls*K + Ks)  # Load elements of Y based on computed indices
    z = x + y # Perform element-wise addition

    # Store
    tl.store(Z + Ls*K + Ks, z)  # Store the result in Z

x, y = arange(4),ones(8, 4)
z = zeros(8, 4)
ya = triton_hello_world[(1,)](x, y, z, 4, 8)
z
print(ya)
print(z)

@triton.jit
def triton_hello_world_block(X, Y, Z, K: tl.constexpr, L: tl.constexpr):
    """
    A Triton kernel for performing element-wise addition on large datasets by dividing the work across multiple blocks.
    
    Args:
    X: A 1D tensor representing the first matrix to add.
    Y: A 2D tensor representing the second matrix to add, divided into blocks.
    Z: A 2D tensor where the result of the addition will be stored, matching the block division of Y.
    K: The size of the inner dimension for X and Y.
    L: The number of elements processed per block.
    
    This function utilizes parallelism by assigning each block a portion of the data to process.
    """
    # Run each program in parallel
    pid = tl.program_id(0)  # Get the program (block) ID
    lid = pid * L  # Compute the local ID based on the block size

    # Use arange to build the shape for loading
    Ks = tl.arange(0, K) # Generate indices for K
    Ls = tl.arange(0, L)[:, None]  # Generate indices for L, reshaped for broadcasting

    # Load from memory
    x = tl.load(X + Ks)  # Load elements of X
    # Load based on program id.
    y = tl.load(Y + (Ls + lid) *K + Ks)  # Load elements of Y for this block
    z = x + y  # Perform element-wise addition

    # Store
    tl.store(Z + (Ls + lid) * K + Ks, z)  # Store the result in Z

L = 2**10 # Set the total length L for the operation.
x, y = arange(4),ones(L, 4) # Create an array 'x' of length 4 and a matrix 'y' of size Lx4 filled with ones. 
z = zeros(L, 4) # Initialize a zero matrix 'z' of the same size as 'y' to store the results.
print(z)
num_blocks = 8 # Define the number of blocks for parallel execution.

# Execute the triton_hello_world_block kernel. The execution is parallelized across (L // num_blocks) blocks.
# It performs an element-wise addition of 'x' and 'y', storing the result in 'z'.
# K=4 represents the length of 'x', and num_blocks defines how many rows of 'y' each program should process.
triton_hello_world_block[(L // num_blocks,)](x, y, z, 4, num_blocks)
print(z.shape, z)

# Constants used throughout
K = 16  # Length of the sequence in each block.
BLOCKS = 8  # Number of blocks.
SEQLEN = K * BLOCKS  # Total length of the sequence to be processed.

# These tensors are prepared for the cumulative sum calculation.
x = arange(SEQLEN) # Create a sequence 'x' from 0 to SEQLEN-1
y = zeros(SEQLEN) # Zero-initialized tensor 'y' of the same length.

def cumsum(x):
    """
    Computes the cumulative sum of a given list 'x'.
    
    Inputs:
    - x: A list or 1D tensor of numerical values.
    
    Outputs:
    - h: The final sum of the list 'x'.
    - y: A list containing the cumulative sum of 'x'.
    """
    y = []  # Initialize the list to store cumulative sums.
    h = 0  # Initialize the cumulative sum variable.
    for k in range(len(x)):  # Iterate over the elements of 'x'.
        h = h + x[k]  # Update the cumulative sum.
        y.append(h)  # Append the current cumulative sum to the list.
    return h, y  # Return the final sum and the list of cumulative sums.


# Compute the cumulative sum of the sequence 'x' using the defined Python function.
h_, y_ = cumsum(x.cpu())
plt.bar(range(SEQLEN), y_) # Plot the cumulative sum using a bar chart.
print(range(SEQLEN), y_)
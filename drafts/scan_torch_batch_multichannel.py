import torch

def blelloch_scan_operator(ca, cb, sa, sb):
    """
    Defines the associative operator:
    (sa, sb) * (ca, cb) = (ca * sa, ca * sb + cb)
    
    Operates in-place for efficiency.
    """
    sa.mul_(ca)  # Element-wise multiplication in-place
    sb.mul_(ca).add_(cb)  # Element-wise multiplication and addition in-place

def blelloch_scan(ca, cb):
    """
    Performs a parallel **exclusive** Blelloch scan over batch, sequence, and channels.

    ca: (B, L, D, N) input sequence (diagonal matrices stored as vectors)
    cb: (B, L, D, N) input sequence (vectors)

    Returns:
    sa: (B, L, D, N) scan result (diagonal matrices stored as vectors)
    sb: (B, L, D, N) scan result (vectors)
    """
    device = ca.device  # Ensure device consistency (CPU or CUDA)
    B, L, D, N = ca.shape  # Batch size, sequence length, channels, vector size

    # Initialize scan outputs on the same device
    sa = torch.ones((B, L, D, N), device=device)  # Identity for diagonal matrices
    sb = torch.zeros((B, L, D, N), device=device)  # Zero vector

    # Upsweep (Reduction phase)
    step = 1
    while step < L:
        indices = torch.arange(step, L, 2 * step, device=device)  # Compute update indices
        sa[:, indices], sb[:, indices] = sa[:, indices - step].clone(), sb[:, indices - step].clone()
        blelloch_scan_operator(ca[:, indices - step], cb[:, indices - step], sa[:, indices], sb[:, indices])
        step *= 2

    # Set the last element to identity and zero for exclusive scan (in-place)
    sa[:, -1].fill_(1.0)  # Identity diagonal matrix
    sb[:, -1].zero_()  # Zero vector

    # Downsweep (Distribution phase)
    step = L // 2
    while step > 0:
        indices = torch.arange(step, L, 2 * step, device=device)
        temp_sa, temp_sb = sa[:, indices - step].clone(), sb[:, indices - step].clone()
        sa[:, indices - step].copy_(sa[:, indices])
        sb[:, indices - step].copy_(sb[:, indices])
        blelloch_scan_operator(temp_sa, temp_sb, sa[:, indices], sb[:, indices])
        step //= 2

    return sa, sb

# Example Usage
B = 2   # Batch size
L = 8   # Sequence length (must be power of 2)
D = 3   # Number of channels
N = 4   # Vector size (size of diagonal matrices)

# Move inputs to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Random diagonal matrices stored as vectors
ca = torch.rand(B, L, D, N, device=device)
cb = torch.rand(B, L, D, N, device=device)

sa, sb = blelloch_scan(ca, cb)

print("Exclusive Scan Results:")
print("sa:", sa.shape)  # Should be (B, L, D, N)
print("sb:", sb.shape)  # Should be (B, L, D, N)

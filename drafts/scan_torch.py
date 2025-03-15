## https://arxiv.org/pdf/2208.04933
## https://arxiv.org/pdf/2312.00752
## https://chatgpt.com/share/67d4fdb0-0d58-800c-9606-9a121149c8c8

import torch

def blelloch_scan_operator(ca, cb, sa, sb):
    """
    Defines the associative operator:
    (sa, sb) * (ca, cb) = (ca * sa, ca * sb + cb)

    ca, sa: (L, N) diagonal matrices stored as vectors
    cb, sb: (L, N) vectors
    """
    new_sa = ca * sa  # Elementwise multiplication (since ca and sa represent diagonal matrices)
    new_sb = ca * sb + cb  # Matrix-vector multiplication simplified
    return new_sa, new_sb

def blelloch_scan(ca, cb):
    """
    Performs an **exclusive** Blelloch scan over sequences of vectors.

    ca: (L, N) input sequence (diagonal matrices stored as vectors)
    cb: (L, N) input sequence (vectors)
    
    Returns:
    sa: (L, N) scan result (diagonal matrices stored as vectors)
    sb: (L, N) scan result (vectors)
    """
    L, N = ca.shape  # Sequence length and vector size

    # Initialize scan outputs
    sa = torch.ones((L, N), device=ca.device)  # Identity for diagonal matrices
    sb = torch.zeros((L, N), device=ca.device)  # Zero vector

    # Upsweep (Reduction phase)
    step = 1
    while step < L:
        for i in range(0, L, 2 * step):
            if i + 2 * step - 1 < L:
                sa[i + 2 * step - 1], sb[i + 2 * step - 1] = blelloch_scan_operator(
                    ca[i + step - 1], cb[i + step - 1], 
                    sa[i], sb[i]
                )
        step *= 2

    # Set the last element to identity and zero for exclusive scan
    sa[-1] = torch.ones(N, device=ca.device)  # Identity diagonal matrix
    sb[-1] = torch.zeros(N, device=ca.device)  # Zero vector

    # Downsweep (Distribution phase)
    step = L // 2
    while step > 0:
        for i in range(0, L, 2 * step):
            if i + 2 * step - 1 < L:
                temp_sa, temp_sb = sa[i + step - 1], sb[i + step - 1]
                sa[i + step - 1], sb[i + step - 1] = sa[i + 2 * step - 1], sb[i + 2 * step - 1]
                sa[i + 2 * step - 1], sb[i + 2 * step - 1] = blelloch_scan_operator(
                    temp_sa, temp_sb, sa[i + 2 * step - 1], sb[i + 2 * step - 1]
                )
        step //= 2

    return sa, sb

# Example Usage
L = 8  # Sequence length (must be power of 2)
N = 4  # Vector size (size of diagonal matrices)

# Random diagonal matrices stored as vectors
ca = torch.rand(L, N)
cb = torch.rand(L, N)

sa, sb = blelloch_scan(ca, cb)

print("Exclusive Scan Results:")
print("sa:", sa)
print("sb:", sb)

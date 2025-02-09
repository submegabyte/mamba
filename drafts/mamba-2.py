import math
import torch
from torch import cfloat
import torch.nn as nn

## rank = 1
def hippo_LegS(n, k):
    if n > k:
        return -(2*n+1)**0.5 * (2*k+1)**0.5
    if n == k:
        return -(n+1)
    return 0

def hippo_LegS_matrix(N):
    A = torch.empty(N, N)

    for i in range(N):
        for j in range(N):
            A[i][j] = hippo_LegS(i, j)

    return A

## Normal Matrix
def hippo_LegS_N_matrix(N):

    P = torch.arange(N).reshape(N, 1)
    P = (P + 1/2)**0.5

    Ah = hippo_LegS_matrix(N)

    AN = Ah + P @ P.T
    return AN

## Diagonal Matrix
## https://chatgpt.com/share/67a68041-935c-800c-b248-37329bbe4afb
def hippo_LegS_D_matrix(N):
    AN = hippo_LegS_N_matrix(N)
    eigenvalues, _ = torch.linalg.eig(AN)

    AD = eigenvalues ## N
    # AD = torch.diag(eigenvalues) ## N x N

    # print(AD)
    # print(AD.shape)
    return AD

## A is an N x N diagonal matrix
# class Discretize2D:

#     ## S4 paper
#     @staticmethod
#     def Bilinear(delta, A, B):
#         N = A.shape[0]
#         I = torch.eye(N)

#         A0 = I - delta / 2 * A
#         A0 = torch.inverse(A0)
#         A1 = I + delta / 2 * A

#         Ad = A0 @ A1
#         Bd = A0 @ (delta * B)

#         return Ad, Bd
    
#     ## DSS paper
#     @staticmethod
#     def ZOH(delta, A, B):
#         N = A.shape[0]
#         I = torch.eye(N)

#         dA = delta * A
#         idA = torch.inverse(dA)
        
#         ## diagonal matrix inverse
#         # idA = 1 / dA

#         dB = delta * B

#         Ad = torch.exp(dA)
#         Bd = idA @ (dA - I) @ dB

#         return Ad, Bd

## A is an N vector
class Discretize1D:
    
    ## DSS paper
    @staticmethod
    def ZOH(delta, A, B):
        ## delta: L x F or B x L x F
        ## A: F x N
        ## B: L x N or Batch x L x N
        F, N = A.shape[-2:]
        # I = torch.eye(N)

        # print(delta.shape, A.shape, B.shape)

        ## diagonal matrix
        dA = delta.unsqueeze(-1) * A.unsqueeze(-3) ## L x F x N or Batch x L x F x N
        # idA = torch.inverse(dA)
        
        ## diagonal matrix inverse
        idA = 1 / dA ## L x F x N or Batch x L x F x N

        deltaS = delta.select(-1, 0) ## L or Batch x L
        deltaS = deltaS.unsqueeze(-1) ## L x 1 or Batch x L x 1
        dB = deltaS * B ## L x N or Batch x L x N
        dBF = dB.unsqueeze(-2) ## L x 1 x N or Batch x L x 1 x N
        dBF = dBF.expand(*dBF.shape[:-2], F, *dBF.shape[-1:]) ## L x F x N or Batch x L x F x N
        # print(deltaS.shape, B.shape, dB.shape)

        # print(delta.shape, A.shape, dA.shape, idA.shape)
        # print(B.shape, deltaS.shape, dB.shape, dBF.shape)

        Ad = torch.exp(dA) ## L x F x N or Batch x L x F x N
        # Bd = idA @ (dA - I) @ dB 
        Bd = (idA * (dA - 1)) * dBF ## L x F x N or Batch x L x F x N

        # print(Ad.shape, Bd.shape)

        return Ad, Bd

# def s4d_kernel(Ad, Bd, C, L):
#     ## Ad is a diagonal matrix (NPLR in s4)
#     ## Cd = C

#     BC = Bd.T * C

#     N = Ad.shape[0]
#     # print(Ad)
#     Ad = torch.diag(Ad)
#     Ad = Ad.unsqueeze(1)  # Convert to column vector
#     exponent = torch.arange(L)

#     print(Ad.shape, exponent.shape)

#     ## vandermonde matrix
#     VAd = Ad ** exponent

#     Kd = BC @ VAd
#     Kd = Kd.real
#     return Kd

class S6(nn.Module):
    def __init__(self, N=3, F=1, delta=1):
        super().__init__()

        ## h'(t) = A h(t) + B x(t)
        ##  y(t) = C h(t)

        ## Before Parameters
        # B = torch.arange(N).reshape(N, 1)
        # B = (2 * B + 1)**0.5

        # P = torch.arange(N).reshape(N, 1)
        # P = (P + 1/2)**0.5

        ## parameters
        self.A = hippo_LegS_D_matrix(N) ## N x
        # print(self.A.dtype)
        self.A = self.A.unsqueeze(0) ## 1 x N
        self.A = self.A.repeat(F, 1) ## F x N
        # print(self.A.shape)
        self.A = nn.Parameter(self.A)
        # self.A = self.A.to(cfloat) ## F x N
        self.sB = nn.Linear(F, N)
        self.sC = nn.Linear(F, N)
        self.sDeltaP = nn.Linear(F, 1)
        ## batched sequence X: L x F
        ## single token x: F
        self.sDelta1 = lambda x: self.sDeltaP(x) ## L x 1 or BL x 1
        self.sDelta = lambda x: self.sDelta1(x).expand(*x.shape[:-1], F) 
        self.tDelta = nn.Softplus() ## L x F or BL x F
        # self.sDelta = lambda x: self.sDeltaP(x) ## L x 1 or BL x 1
        # self.tDelta = nn.Softplus() ## L x 1 or BL x 1

        ## scalars
        self.N = N ## state size
        self.F = F ## feature embedding length
        self.delta = delta ## step size
        
    ## x: B x L x D
    def forward(self, x):
        # h = torch.zeros(self.N)

        # L = x.shape[0]
        L, F = x.shape[-2:]
        
        xv = x.view(-1,F) ## L x F or BL x F

        ## variable dimension view and select
        ## https://chatgpt.com/share/67a7cbfb-3130-800c-985d-ae2ffc1483d9

        B = self.sB(xv) ## L x N or BL x N
        B = B.view(*x.shape[:-1], self.N) ## L x N or Batch x L x N

        C = self.sC(xv) ## L x N or BL x N
        C = C.to(cfloat)
        C = C.view(*x.shape[:-1], self.N) ## L x N or Batch x L x N

        delta = self.tDelta(self.delta + self.sDelta(xv)) ## L x F or BL x F
        delta = delta.view(*x.shape[:-1], F)
        delta = delta.to(cfloat) ## L x F or Batch x L x F

        Ad, Bd = Discretize1D.ZOH(delta, self.A, B) ## L x F x N or Batch x L x F x N
        # Kd = s4d_kernel(Ad, Bd, self.C, L)

        # print(Kd.shape, x.shape)
        
        # x = Kd @ x

        h = torch.zeros(*Ad.shape[:-3], *Ad.shape[-2:]) ## F x N or Batch x F x N

        for i in range(L):
            ## https://chatgpt.com/share/67a7cbfb-3130-800c-985d-ae2ffc1483d9
            xi = x.select(-2, i) ## F or Batch x F

            Adi = Ad.select(-3, i) ## F x N or Batch x F x N
            Bdi = Bd.select(-3, i) ## F x N or Batch x F x N

            # print(Ad.shape, Bd.shape)
            # print(Adi.shape, Bdi.shape)
            # print(h.shape, xi.shape)

            h = Adi * h + Bdi * xi.unsqueeze(-1) ## F x N or Batch x F x N
        
        Cl = C.select(-2, -1) ## N or Batch x N
        Clu = Cl.unsqueeze(-2) ## 1 x N or Batch x 1 x N
        Cle = Clu.expand(*Clu.shape[:-2], F, *Clu.shape[-1:]) ## F x N or Batch x F x N

        # print(Cle.shape, h.shape)
        # print(Cle.dtype, h.dtype)

        # print(h)

        ## https://chatgpt.com/share/67a7e6ce-71c8-800c-af23-747f5fdc7651
        y = torch.einsum('...i,...i->...', Cle, h)   ## F or Batch x F

        # print(y)

        return y


if __name__ == "__main__":
    L = 200
    F = 4 ## embedding
    N = 64 ## state

    x = torch.randn(L, F) ## L x 1
    model = S6(N, F) ## L x 1

    y = model(x)

    print(y.shape) ## F or Batch x F
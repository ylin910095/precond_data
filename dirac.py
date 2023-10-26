import numpy as np
from gamma import gamma_factory
import torch

class Dirac_Matrix:
    def  __init__(self, U, kappa):
        """
        Arguments:
            U: the input gauge field with shape (batch_size, n_dim = 2, X, T)
            kappa: the hopping parameter     
        """
        self.n_dim = 2

        # The index of the time direction for applying the antiperiodic boundary condition
        # Need to be consistent with the gamma matrices definition
        self.time_index = 2

        self.U = U
        self.kappa = kappa
        
        assert torch.is_complex(U)
        assert len(U.shape) == 4
        assert U.shape[1] == self.n_dim
        
        self.lattice_shape = U.shape[2:]
        self.gammas = np.stack(gamma_factory(n_dim=self.n_dim, incl_g5=False), 
                               axis=0)
        self.gammas = torch.from_numpy(self.gammas).type_as(U)
        self.gamma5 = 1j * self.gammas[0] @ self.gammas[1]
        self.identity = torch.zeros_like(self.gammas[0]) # identity matrix
        for i in range(self.n_dim):
            self.identity[i,i] = 1 
    
    def apply(self, x, dagger=False, gamma5=False):
        """
        Argument:
            x: the quark field vector with shape [batch_size, X, T, n_dim]
               or [X, T, n_dim]
            dagger: apply the D opertor if False else apply 
                    the conjugate transpose of the D operator.
                    This is implemented using 
                    D_dagger = gamma5 @ D @ gamma5
        Returns: 
            y: the matrix-vector product y=Dx where D is the Dirac matrix.
        """
        x = x.type_as(self.U)
        x = self._normalize_vector_shape(x)
        if dagger:
            x = torch.einsum("ij, ...j->...i", self.gamma5, x)
        
        padded_x, slice_idx = self._antiperiodic_pad_vector(x)
        H = torch.zeros_like(x)
        for mu in range(self.n_dim):
            forward_x = torch.roll(padded_x, shifts=-1, dims=1+mu)
            forward_x = forward_x[slice_idx]
            forward_U = self.U[:, mu].unsqueeze(-1) # dummy spinor dimension
            forward_x = forward_U * forward_x

            backward_x = torch.roll(padded_x, shifts=1, dims=1+mu)
            backward_x = backward_x[slice_idx]
            backward_U = torch.roll(self.U[:, mu], shifts=1, dims=1+mu).conj()
            backward_U = backward_U.unsqueeze(-1) # dummy spinor dimension
            backward_x = backward_U * backward_x
            
            H = H + torch.einsum("ij, ...j->...i", self.identity - self.gammas[mu], forward_x)
            H = H + torch.einsum("ij, ...j->...i", self.identity + self.gammas[mu], backward_x)          
        y = x - (self.kappa * H)
        if dagger:
            y = torch.einsum("ij, ...j->...i", self.gamma5, y)
        
        return y
    
    def _normalize_vector_shape(self, x):
        x = x.clone() # make a copy first
        assert len(x.shape) in [3, 4], f'unknown vector shape {x.shape}'
        if x.shape == 3:
            x = x.unsqueeze(0) # dummy batch dimension
        assert x.shape[-1] == self.n_dim
        assert x.shape[1:-1] == self.lattice_shape
        return x
    
    def _antiperiodic_pad_vector(self, x):
        """
        Pad vector x with antiperiodic boundary condition 
        in time

        Returns padded vector and the slice indices
        needed to recover the unpadded vector
        """
        padded_x = x.clone()

        s1 = [slice(None), ]*len(x.shape)
        s1[self.time_index] = slice(0, 1)
        forward_pad = -padded_x.clone()[s1]

        s2 = [slice(None), ]*len(x.shape)
        s2[self.time_index] = slice(x.shape[self.time_index]-1, 
                                    x.shape[self.time_index])
        backward_pad = -padded_x.clone()[s2]
    
        padded_x = torch.cat([backward_pad, padded_x, forward_pad], 
                              dim=self.time_index)
        
        # Slice indices needed to recover the unpadded vector
        slice_idx = [slice(None), ] * len(padded_x.shape)
        slice_idx[self.time_index] = slice(1, -1)
        
        assert torch.allclose(padded_x[slice_idx], x), 'invalid padding'
        return (padded_x, tuple(slice_idx))
    
def example():
    kappa = 0.276
    test_U = np.load("config.l8-N200-b2.0-k0.276-unquenched-test.x.npy")
    test_U = torch.from_numpy(test_U)
    test_U = torch.exp(1j*test_U).to(torch.cdouble) # dont forget to take the exponential!
    test_U = test_U[:1] # just take one configuration
    D = Dirac_Matrix(test_U, kappa=kappa)

    x = torch.zeros([1,8,8,2])
    x = torch.arange(8*8*2).reshape(1,8,8,2)
    y = D.apply(D.apply(x),dagger=True) # y = (D^dagger D)x 

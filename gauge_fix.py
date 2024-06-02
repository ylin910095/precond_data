import sys
from collections.abc import Iterable
import torch
from torch import Tensor
import numpy as np

def step_gf(links: Tensor,
            k_field: Tensor,
            omega_tilde_dagger: Tensor) -> Tensor:
    """
    Take an iterative gauge-fix step
    """
    lattice_size = links.shape[2:]
    nd = len(lattice_size)
    y = omega_tilde_dagger

    for mu in range(nd):
        forward_links = links[:, mu]
        forward_omg = torch.roll(omega_tilde_dagger, shifts=-1, dims=1+mu)
        forward_k = k_field[:, mu]
        backward_links = torch.roll(links[:, mu], shifts=1, dims=1+mu).conj()
        backward_omg = torch.roll(omega_tilde_dagger, shifts=1, dims=1+mu)
        backward_k = torch.roll(k_field[:, mu], shifts=1, dims=1+mu)
        y = y + (forward_links * forward_omg * forward_k) + \
                (backward_links * backward_omg * backward_k)

    return y

def gauge_transform_dagger(links, omega_dagger):
    """
    Apply the gauge transformation specified by omega^dagger to the SU(N) field
    """
    lattice_size = links.shape[2:]
    nd = len(lattice_size)
    out = torch.empty_like(links)
    for mu in range(nd):
        out[:, mu] = (omega_dagger).conj() *\
              links[:,mu] * torch.roll(omega_dagger, shifts=-1, dims=1+mu)
    return out

def iterative_gf(
    links: Tensor,
    k_field: Tensor,
    n_iter: int|None = None,
    origin: Iterable[int]|None = None,
    eps: float = 1e-15) -> Tensor:
    """
    Gauge fix the link field according to the k-field by performing N gauge-fix
    iterations.
    """
    # Set the initial trial iterations
    nd = links.shape[1]
    lattice_size = links.shape[2:]
    assert nd == len(lattice_size), f"{nd = } not compatible with {links.shape = }"

    k_field = k_field.squeeze().unsqueeze(0)

    # Worst case situation
    if n_iter is None:
        n_iter = sum(lattice_size)*2

    # Intialize the source field and remove the unneeded dimension
    omega_dagger_tilde = torch.zeros_like(links[:, 0])
    if origin is None:
        origin = [0, ]*nd
    sindx = (slice(None), *origin)
    omega_dagger_tilde[sindx] = 1

    def _normalize(omg):
        # abs is important to keep the determinant positive such that
        norm = torch.clamp(torch.abs(omg), min=1)
        return omg/(norm.detach()) # we don't want the normalization affecting the gradient

    for _ in range(n_iter):
        omega_dagger_tilde = step_gf(links, k_field, omega_dagger_tilde)
        omega_dagger_tilde = _normalize(omega_dagger_tilde)

    # Project to unitarity - add small epsilon such that the zero
    omega_dagger_tilde = omega_dagger_tilde + eps
    omega_dagger = omega_dagger_tilde / omega_dagger_tilde.abs()
    links_gf = gauge_transform_dagger(links, omega_dagger)
    return links_gf

def compute_plaq(links: Tensor, mu: int, nu: int) -> Tensor:
    """Compute the field of plaquettes from U(1) links"""
    plaq = links[:, mu] * torch.roll(links[:,nu], -1, dims=1+mu) *\
           torch.roll(links[:, mu], -1, 1+nu).conj() * links[:,nu].conj()
    return plaq

if __name__ == "__main__":
   data, save = sys.argv[1:]

   data = np.load(data)
   data = torch.as_tensor(data)
   data = torch.exp(1j * data)

   lattice_size = data.shape[2:]
   print(f"{lattice_size = }")
   assert len(lattice_size) == 2, "only support 2d"

   fix_k = torch.zeros(2, *lattice_size, dtype=data.real.dtype)
   fix_k[0, :-1] = 1
   fix_k[1, 0, :-1] = 1

   gf = iterative_gf(data, fix_k)

   # Check gauge-fixed configs to see if they are correct
   isone = torch.isclose(gf, torch.ones_like(gf)).to(fix_k.dtype)
   idx = torch.where(fix_k == 1)
   bidx = (slice(None), *idx) # dummy batch
   assert torch.allclose(isone[bidx], fix_k[idx])

   # Check plaq to make sure they are unchanged
   plaq_before = compute_plaq(data, mu=0, nu=1)
   plaq_after = compute_plaq(gf, mu=0, nu=1)
   assert torch.allclose(plaq_before, plaq_after)
   print(f"{plaq_before.mean() = }, {plaq_after.mean() = }")

   # Find the angles
   gf = -1j*torch.log(gf)
   assert torch.allclose(gf.imag, torch.zeros_like(gf.imag))
   gf = gf.real

   np.save(save, gf.detach().numpy())

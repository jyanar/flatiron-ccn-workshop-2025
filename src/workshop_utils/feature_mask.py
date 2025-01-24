#!/usr/bin/env python3

import numpy as np
from nemos.basis._basis import Basis, AdditiveBasis
from typing import Literal, Union, Callable, List
from numpy.typing import NDArray


BLOCK_STRUCTURE_TYPES = Union[Literal['all', 'none', 'self'], Callable]

__all__ = ["create_feature_mask", "create_feature_mask_paramgrid"]

def create_feature_mask(design_matrix_basis: Basis,
                        block_structure: Union[List[BLOCK_STRUCTURE_TYPES], BLOCK_STRUCTURE_TYPES] = 'all',
                        n_neurons: int = 1) -> NDArray:
    """Helper function to create boolean feature mask.

    The feature mask is an array consists of 0s and 1s defining which features will be
    included in the model. It has shape ``(n_features, n_neurons)``, where
    ``n_features`` is determined using ``design_matrix_basis`` and ``n_neurons`` is
    given by the user. The feature mask consists of several "blocks", each of which is
    an experimentally-relevant set of features (e.g., spikes, behavior) and is defined
    by the ``design_matrix_basis``.

    ``block_structure`` possible values:

    - ``'all'``: all ones

    - ``'none'``: all zeroes

    - ``'self'``: self-connectivity, i.e., ``np.repeat(np.eye(block_n_neurons),
      n_basis_funcs)``

    - function: a function that accepts the arguments ``n_rows, n_columns`` and returns
      an array of 0s and 1s with shape ``(n_rows, n_columns)``.

    Parameters
    ----------
    design_matrix_basis :
        The basis that creates the design matrix. This is likely an AdditiveBasis which
        combines several other bases and is how we determine the block structure.
    block_structure :
        Structure of each block, see above for details.
    n_neurons :
        The number of neurons for which the mask will be used.

    Returns
    -------
    feature_mask :
        The feature mask to use.

    """
    mask = np.zeros((design_matrix_basis.n_output_features, n_neurons))
    # replace with new iter method
    block_sizes = [b.n_output_features for b in design_matrix_basis._iterate_over_components()]
    if isinstance(block_structure, str):
        block_structure = len(block_sizes) * [block_structure]
    if len(block_sizes) != len(block_structure):
        raise ValueError("block_structure length not the same as the number of blocks defined by design_matrix_basis!"
                        f" len(block_structure): {len(block_structure)}, len(design_matrix_basis): {len(block_sizes)}")
    if any([b not in ["all", "none", "self"] and not callable(b) for b in block_structure]):
        bad_vals = set([b for b in block_structure if b not in ["all", "none", "self"] and not callable(b)])
        raise ValueError(f"block_structure values must be one of 'all', 'none', 'self' or a function, but got {bad_vals}!")
    shape_sum = 0
    for s, b in zip(block_sizes, block_structure):
        if b == "all":
            mask[shape_sum:shape_sum+s] = np.ones((s, n_neurons))
        elif b == "self":
            mask[shape_sum:shape_sum+s] = np.repeat(np.eye(n_neurons), s//n_neurons, 0)
        elif callable(b):
            m = b(s, n_neurons)
            if (m.astype(bool).astype(float) != m).any():
                raise ValueError(f"block_structure callable {b} must return a binary array!")
            mask[shape_sum:shape_sum+s] = m
        shape_sum += s
    return mask


def create_feature_mask_paramgrid(basis: AdditiveBasis,
                                  basis1_n_basis_funcs: List[int],
                                  basis2_n_basis_funcs: List[int],
                                  n_neurons: int):
    param_grid = []
    # include all position (basis1), exclude all speed (basis2)
    for b1 in basis1_n_basis_funcs:
        basis.basis1.n_basis_funcs = b1
        basis.basis2.n_basis_funcs = basis2_n_basis_funcs[0]
        param_grid.append({"glm__feature_mask": [create_feature_mask(basis, ["all", "none"], n_neurons=n_neurons)],
                           "basis__basis1__n_basis_funcs": [b1], "basis__basis2__n_basis_funcs": [basis2_n_basis_funcs[0]]})

    # include all speed, exclude all position
    for b2 in basis2_n_basis_funcs:
        basis.basis2.n_basis_funcs = b2
        basis.basis1.n_basis_funcs = basis1_n_basis_funcs[0]
        param_grid.append({"glm__feature_mask": [create_feature_mask(basis, ["none", "all"], n_neurons=n_neurons)],
                           "basis__basis1__n_basis_funcs": [basis1_n_basis_funcs[0]], "basis__basis2__n_basis_funcs": [b2]})

    # exclude all of both
    for b1 in basis1_n_basis_funcs:
        for b2 in basis2_n_basis_funcs:
            basis.basis1.n_basis_funcs = b1
            basis.basis2.n_basis_funcs = b2
            param_grid.append({"glm__feature_mask": [create_feature_mask(basis, "all", n_neurons=n_neurons)],
                               "basis__basis1__n_basis_funcs": [b1], "basis__basis2__n_basis_funcs": [b2]})

    return param_grid

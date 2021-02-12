"""
Matrix operation and allocation functions.

This file contains the functions that perform matrix allocation and operations.

This file can also be imported as a module and contains the following
functions:


"""
# module imports
import numpy as np
from typing import Tuple, Union
from scipy.sparse import hstack, vstack, coo_matrix


# functions and classes
# def finite_difference_mat(npars: int) -> np.ndarray:
#     """Create the finite difference matrix for regularization."""
#     fdmat = np.zeros((int(npars - 1), npars))
#     for i in range(int(npars - 1)):
#         fdmat[i, i] = -1
#         fdmat[i, i + 1] = 1
#     return fdmat


def roughness(m: np.ndarray) -> float:
    """
    A function to compute the 'roughness' of a model data array.
    Roughness is defined as the L2 norm of the 2nd order differential.
    The finite difference to compute approximate a differential.

    Parameters
    ----------
    m : np.ndarray
        The model array.

    Returns
    -------
    float
        The value of roughness for the model.
    """
    return np.linalg.norm(np.diff(m, 2), 2)


def mse(d: np.ndarray, d_pred: np.ndarray) -> float:
    """
    A function to compute the mean squared error between recorded and predicted
    data arrays.

    Parameters
    ----------
    d : np.ndarray
        The recorded data array (observations).

    d_pred : np.ndarray
        The predicted data from the model.

    Returns
    -------
    float
        The value of roughness for the model.
    """
    r = d - d_pred
    return np.mean(r**2)


def finite_difference_mat(npars: int) -> np.ndarray:
    """Create the finite difference matrix for regularization."""
    fdmat = np.zeros((npars, npars))
    for i in range(int(npars - 1)):
        fdmat[i, i] = -1
        fdmat[i, i + 1] = 1
    return fdmat


def embed_mat(a: np.ndarray,
              b: np.ndarray,
              rs: int = 0,
              cs: int = 0,
              ) -> np.ndarray:
    """
    Embeds a smaller matrix inside a bigger one

    """
    if a.shape >= b.shape:
        raise ValueError("a must be smaller in dimensions than b.")

    a, b = a.copy(), b.copy()

    b[rs:rs + a.shape[0], cs:cs + a.shape[1]] = a

    return b


def const_constraint_coeffs(model_indices: np.ndarray,
                            unique_labels: np.ndarray,
                            label: np.ndarray,
                            ) -> Tuple[np.ndarray,
                                       np.ndarray,
                                       np.ndarray,
                                       ]:
    if label not in unique_labels:
        raise ValueError(f"{label} not in {unique_labels}.")

    col = model_indices[unique_labels == label]

    return (np.array([0, ]).astype(int),
            col.astype(int),
            np.array([1, ])
            )


def sum_constraint_coeffs(model_indices: np.ndarray,
                          **kwargs,
                          ) -> Tuple[np.ndarray,
                                     np.ndarray,
                                     np.ndarray,
                                     ]:

    cols = model_indices.flatten()

    return (np.zeros(len(cols)).astype(int),
            cols.astype(int),
            np.ones(len(cols))
            )


def get_interpolation_coeffs(labels: np.ndarray,
                             nodes: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function computes the the design matrix coefficients for a pair of
    adjacent nodes by its linear interpolation from the label to either node.

    Parameters
    ----------
    labels : np.ndarray
        A numpy array containing the data labels to obtain interpolation
        coefficients for (factor between 0 - 1).

    nodes : np.ndarray
        A numpy array of a pair of adjacent node values with index (i) and i+1,
        to interpolate between.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A pair of numpy arrays of interpolation coefficients for a pair
        of adjacent nodes with index i, i+1.

    Examples
    --------
    >>> labels = np.array([1, 2.5, 5])  # the data labels
    >>> nodes = np.array([1, 5])  # the node pairs to linearly interpolate to
    >>> get_interpolation_coeffs(labels, nodes)
    (array([1.   , 0.625, 0.   ]), array([0.   , 0.375, 1.   ]))
    """

    ab = (nodes[1] - labels) / (nodes[1] - nodes[0])
    aa = 1 - ab

    return ab, aa


def build_interp_coeffs_as_triplet(labels: np.ndarray,
                                   nodes: np.ndarray,
                                   c: int = 1
                                   ) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray
                                              ]:
    """
    This functions allocates the coefficients (value) and location (row and
    column) to three numpy arrays. This choice is inspired by the COOrdinate
    (or 'ijv' / 'triplet') format, which is a sparse matrix format. As such
    the output is designed to work directly with scipy.sparse.coo_matrix to
    save on memory requirements, before the matrix is compressed. The row
    index values are just allocated in the order that they appear in the passed
    label array. This assumes the rows are sorted so that each row corresponds
    to correct data value in a separate data array. If it does not, you must
    sort them first, otherwise it will be out of order.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html

    Parameters
    ----------
    labels : np.ndarray
        A numpy array containing the data labels to obtain interpolation
        coefficients for (factor between 0 - 1).

    nodes : np.ndarray
        A numpy array of two values (nodes) to interpolate between.
        for.

    c : int [-1 or 1]
        An interger value of -1 or 1 that is multiplied by the coefficients
        to make them positive or negative in accordance with the equation.

    Examples
    --------
    """
    cols, rows, vals = np.array([], dtype=int), np.array([], dtype=int),\
        np.array([], dtype=float)

    for n in range(len(nodes[:-1])):
        nb, na = n, n + 1  # unpack adjacent node distance index

        inds = np.where((labels > nodes[nb]) & (labels <= nodes[na]))

        # assume rows are consistent between data and labels and therefore
        # we can use the indices from the output of np.where.
        rows_b, rows_a = inds[0], inds[0]
        # closest nodal edge that is less than values between nodes n, n+1.
        # we need to save the column indices as is required for the COO format.
        cols_b = np.array([n for _ in inds[0]])
        # closest nodal edge that is >= distances between nodes n, n+1.
        cols_a = (cols_b + 1)
        # compute interpolation coefficients for each label.
        ab, aa = get_interpolation_coeffs(labels[inds], [nodes[nb], nodes[na]])

        # append to numpy array
        rows = np.append(rows, np.append(rows_b, rows_a))
        cols = np.append(cols, np.append(cols_b, cols_a))
        vals = np.append(vals, np.append(c * ab, c * aa))

    return rows.astype(int), cols.astype(int), vals


def build_constant_coeffs_as_triplet(unique_indices: np.ndarray,
                                     unique_labels: np.ndarray,
                                     raw_labels: np.ndarray,
                                     coeff_const: int = 1
                                     ) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray
                                                ]:
    """
    This functions allocates the coefficients (value) and location (row and
    column) to three numpy arrays. This choice is inspired by the COOrdinate
    (or 'ijv' / 'triplet') format, which is a sparse matrix format. As such
    the output is designed to work directly with scipy.sparse.coo_matrix to
    save on memory requirements, before the matrix is compressed. The row
    index values are just allocated in the order that they appear in the passed
    label array. This assumes the rows are sorted so that each row corresponds
    to correct data value in a separate data array. If it does not, you must
    sort them first, otherwise it will be out of order.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html

    Parameters
    ----------

    unique_labels : np.ndarray
        A numpy array containing the unique and ordered constant data labels,
        e.g., np.array(['IW.REDW', 'MB.BUT', 'US.DUG', 'UU.TCU', 'WY.YUF'])
    nodes : np.ndarray
        A numpy array of two values (nodes) to interpolate between.
        for.

    c : int [-1 or 1]
        An interger value of -1 or 1 that is multiplied by the coefficients
        to make them positive or negative in accordance with the equation.

    Examples
    --------
    """

    rows = np.array(range(len(raw_labels)))
    cols = np.select([raw_labels == label for label in unique_labels],
                     unique_indices
                     )
    vals = (np.zeros(len(raw_labels)) + coeff_const)

    return rows.astype(int), cols.astype(int), vals


def compress_matrices(G: Union[np.ndarray, coo_matrix],
                      d: Union[np.ndarray, coo_matrix],
                      ) -> Tuple[coo_matrix, coo_matrix]:

    GTG = coo_matrix(np.dot(G.T, G))
    GTd = coo_matrix(np.dot(G.T, d))

    return GTG, GTd


def apply_constraints(GTG: coo_matrix,
                      GTd: coo_matrix,
                      F: coo_matrix,
                      h: coo_matrix) -> coo_matrix:

    # This code block sets up the matrix to solve for constraints exactly by
    # ... the method of lagrange multipliers.
    coeffs_top_row = hstack((GTG, F.T))

    residual = np.sum(np.array(coeffs_top_row.shape) - np.array(GTG.shape))
    bottom_padding = np.zeros((residual, residual))

    coeffs_bottom_row = hstack((F, bottom_padding))

    GTGcons = vstack((coeffs_top_row, coeffs_bottom_row))

    GTdcons = vstack((GTd, h))

    return GTGcons, GTdcons

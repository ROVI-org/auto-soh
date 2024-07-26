""" Utility definitions for base estimators """
import numpy as np


def ensure_positive_semi_definite(Sig: np.ndarray) -> np.ndarray:
    """
    Function to ensure the matrix is positive semi-definite. If the matrix is positive semi-definite, it simply returns
    the original matrix. Otherwise, it returns the "closest" positive semidefinite matrix to the original one provided.

    Args:
        Sig: matrix to check for positive semi-definiteness

    Returns:
        "Closest" (most similar) positive semi-definite matrix
    """
    if np.allclose(Sig, Sig.T):  # checking if it is symmetric
        try:
            # checking for positive semi-definiteness
            np.linalg.cholesky(Sig)  # throws LinAlgError if not
            return Sig.copy()
        except np.linalg.LinAlgError:
            return enforce_positive_semi_defiteness(Sig)
    else:
        return enforce_positive_semi_defiteness(Sig)


def enforce_positive_semi_defiteness(Sig: np.ndarray) -> np.ndarray:
    """
    Finds nearest positive semi-definite matrix to the one provided.

    Ref.: Nicholas J. Higham, “Computing a Nearest Symmetric Positive
    Semidefinite Matrix,” Linear Algebra and its Applications, 103,
    103–118, 1988

    Args:
        Sig: matrix that should be positive semi-definite but isn't
    """
    # Perform singular value decomposition
    _, S_diagonal, V_complex_conjugate = np.linalg.svd(Sig)
    H_matrix = np.matmul(V_complex_conjugate.T, np.matmul(S_diagonal, V_complex_conjugate))
    return (Sig + Sig.T + H_matrix + H_matrix.T) / 4

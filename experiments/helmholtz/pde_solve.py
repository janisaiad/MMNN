import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_helmholtz_dirichlet(
    k2: np.ndarray,
    f: np.ndarray,
    domain_mask: np.ndarray,
    boundary_mask: np.ndarray,
    g: np.ndarray,
    h: float,
) -> np.ndarray:  # we solve (-Δ - k^2)u = f with u=g on the boundary on a masked grid #
    n = int(k2.shape[0])  # we read grid size #
    if k2.shape != (n, n) or f.shape != (n, n) or g.shape != (n, n):  # we validate shapes #
        raise ValueError("k2, f, g must be (n,n)")  # we fail loudly #
    if domain_mask.shape != (n, n) or boundary_mask.shape != (n, n):  # we validate shapes #
        raise ValueError("domain_mask, boundary_mask must be (n,n)")  # we fail loudly #
    dm = domain_mask.astype(bool)  # we cast #
    bm = boundary_mask.astype(bool)  # we cast #
    if not np.all(bm <= dm):  # we validate boundary subset #
        raise ValueError("boundary_mask must be subset of domain_mask")  # we fail loudly #

    idx = -np.ones((n, n), dtype=np.int64)  # we allocate index map #
    pts = np.argwhere(dm)  # we list domain points #
    for p, (i, j) in enumerate(pts):  # we map to contiguous indices #
        idx[int(i), int(j)] = int(p)  # we assign #
    m = int(pts.shape[0])  # we store number of unknowns #
    if m == 0:  # we guard empty domain #
        return np.zeros((n, n), dtype=np.float64)  # we return zeros #

    A = sp.lil_matrix((m, m), dtype=np.float64)  # we allocate sparse matrix #
    b = np.zeros(m, dtype=np.float64)  # we allocate rhs #
    inv_h2 = 1.0 / float(h * h)  # we compute 1/h^2 #

    for p, (i0, j0) in enumerate(pts):  # we fill system #
        i = int(i0)  # we cast #
        j = int(j0)  # we cast #
        if bm[i, j]:  # we impose dirichlet on boundary #
            A[p, p] = 1.0  # we set identity #
            b[p] = float(g[i, j])  # we set value #
            continue  # we skip stencil #

        diag = 0.0  # we accumulate diagonal #
        rhs = float(f[i, j])  # we set forcing #
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  # we loop neighbors #
            ii = i + int(di)  # we compute #
            jj = j + int(dj)  # we compute #
            if ii < 0 or ii >= n or jj < 0 or jj >= n:  # we treat outside grid as dirichlet zero #
                rhs = rhs + inv_h2 * 0.0  # we add nothing #
                diag = diag + inv_h2  # we add diagonal #
                continue  # we continue #
            if not dm[ii, jj]:  # we treat outside domain as dirichlet zero #
                rhs = rhs + inv_h2 * 0.0  # we add nothing #
                diag = diag + inv_h2  # we add diagonal #
                continue  # we continue #
            q = int(idx[ii, jj])  # we lookup neighbor #
            A[p, q] = A[p, q] - inv_h2  # we add off-diagonal #
            diag = diag + inv_h2  # we add diagonal #

        A[p, p] = A[p, p] + 4.0 * inv_h2 - float(k2[i, j])  # we set diagonal for (-Δ - k^2) #
        b[p] = rhs  # we set rhs #

    A_csr = A.tocsr()  # we convert for solver #
    u_vec = spla.spsolve(A_csr, b)  # we solve sparse system #
    u = np.zeros((n, n), dtype=np.float64)  # we allocate full grid #
    for p, (i0, j0) in enumerate(pts):  # we scatter back #
        u[int(i0), int(j0)] = float(u_vec[int(p)])  # we write #
    u[~dm] = 0.0  # we zero outside domain #
    return u  # we return solution #


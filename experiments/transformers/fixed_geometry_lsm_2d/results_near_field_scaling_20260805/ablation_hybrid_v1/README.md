# Archived hybrid-PCG ablation

This recoverable archive contains the interrupted first hybrid sweep.  That
variant allowed a log-gain correction radius of 2 and regularized the total
preconditioner gain toward the identity.  It was replaced in the reported grid
because this unfairly penalized the analytic angular-Jacobi base.

The reported hybrid instead bounds the learned residual correction to a
log-radius of 0.5 and regularizes that correction relative to the analytic
base.  The archived rows and checkpoints are not loaded by the final analyzer.

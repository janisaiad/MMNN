graph TD
    %% NODES
    Input[<b>Assumption 1: Input X</b><br>i.i.d. Gaussian / Indep. Rows]
    
    Conc[<b>Lemma 3.1 & 2.1</b><br>Concentration of Measure<br>(Hanson-Wright / Lipschitz)]
    
    TensorCalc[<b>Lemma 3.6 & 5.1</b><br>Fourth Moment Calculation<br>Decorrelating σ(Wx)]
    
    TensorLimit[<b>Theorem 4.2</b><br>Tensor Reduction<br>Reduces matrix to Tensor T]
    
    Resolvent[<b>Appendix A / Prop A.3</b><br>Resolvent Expansion &<br>Leave-One-Out Bounds]
    
    Thm26[<b>Theorem 2.6</b><br>General Limit for A ⊙ B<br>Free Mult. Convolution]
    
    Thm27[<b>Theorem 2.7 (Result)</b><br>NTK Spectrum<br>MP ⊠ Deterministic]

    %% EDGES
    Input -->|Provides sub-Gaussianity| Conc
    Input -->|Allows integration by parts| TensorCalc
    
    Conc -->|Bounds error terms| Resolvent
    Conc -->|Bounds variance| TensorLimit
    
    TensorCalc -->|Defines the shape of T| TensorLimit
    
    Resolvent -->|Proves Stieltjes convergence| Thm26
    TensorLimit -->|Identifies the limiting object| Thm26
    
    Thm26 -->|Applied to NTK structure| Thm27
    Lemma53[<b>Lemma 5.3</b><br>Eigenvalues of Tensor Q] -->|Explicit values| Thm27

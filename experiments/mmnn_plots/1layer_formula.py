import jax.numpy as jnp # we import jax.numpy for numerical operations
import numpy as np # we import numpy for compatibility with scipy
from scipy.stats import norm, multivariate_normal # we import norm and multivariate_normal from scipy for statistical functions
import matplotlib.pyplot as plt # we import matplotlib for plotting
from tqdm import tqdm # we import tqdm for the progress bar

def expectation_relu(mu, sigma): # we define the function for the relu expectation
    """
    we calculate e[relu(x)] for x ~ n(mu, sigma^2).
    """
    if sigma == 0: # we ensure sigma is not zero to avoid division by zero
        return max(0, mu)
    
    phi = norm.pdf # we define phi as the probability density function
    Phi = norm.cdf # we define phi as the cumulative distribution function
    z = mu / sigma # we define the standardized value z
    return sigma * phi(z) + mu * Phi(z) # we return the expected value

def calculate_riemann_integral(mu1, sigma1, mu2, sigma2, rho, num_steps=50): # we define the function for the integral calculation
    """
    we calculate the integral ∫i₁₁(σ₁₂)dσ₁₂ using a riemann sum.
    this represents the covariance term cov(relu(x₁), relu(x₂)).
    this method is for educational purposes and is less efficient than an algebraic calculation.
    """
    if rho == 0: # we handle the case where rho is zero, the covariance is null
        return 0.0

    rho_steps = jnp.linspace(0, rho, num_steps) # we create a sequence of rho values from 0 to the target rho
    if num_steps > 1: # we ensure there is more than one step
        delta_rho = rho / (num_steps - 1) # we calculate the width of each small rho interval
    else: # otherwise
        delta_rho = rho # the width is rho itself
    
    delta_sigma12 = sigma1 * sigma2 * delta_rho # we calculate the differential of the covariance
    
    integral_sum = 0.0 # we initialize the sum for the integral
    
    for rho_i in rho_steps: # we loop over each small step of the correlation
        # we calculate the covariance matrix for the current rho_i
        cov_matrix = [[sigma1**2, rho_i * sigma1 * sigma2],
                      [rho_i * sigma1 * sigma2, sigma2**2]]
        
        try: # we use a try-except block to handle calculation errors
            # we create a multivariate normal distribution
            mvn = multivariate_normal(mean=[mu1, mu2], cov=cov_matrix, allow_singular=True)
            # we calculate p(x₁ > 0, x₂ > 0)
            prob_both_positive = mvn.cdf([0, 0])
        except np.linalg.LinAlgError: # we catch linear algebra errors
            prob_both_positive = 0.0 # in case of error, we set the probability to 0

        # we add the area of the current riemann rectangle to the sum
        integral_sum += prob_both_positive * delta_sigma12

    return integral_sum # we return the sum of the integral

def expectation_product_relu(mu1, sigma1, mu2, sigma2, rho, num_steps=50): # we define the function for the expectation of the product of relus
    """
    we calculate e[relu(x₁)relu(x₂)] using numerical integration for the covariance term.
    """
    # we calculate the covariance term with the riemann sum
    cov_term = calculate_riemann_integral(mu1, sigma1, mu2, sigma2, rho, num_steps)
    
    # we calculate the expectation of each relu individually
    e_relu1 = expectation_relu(mu1, sigma1)
    e_relu2 = expectation_relu(mu2, sigma2)
    
    # the expectation of the product is the sum of the covariance and the product of the expectations
    return cov_term + e_relu1 * e_relu2

def calculate_ntk_expectation_over_b(beta, sigma1, sigma2, rho, num_b_steps=20, num_rho_steps=50):
    """
    we calculate the expectation of the product of relus, averaged over a gaussian variable b.
    e_b[e[relu(x1)relu(x2)|b]] where mu1=mu2=beta*b.
    this is computed using a riemann sum over b.
    """
    b_values = np.linspace(-4, 4, num_b_steps) # we define the range for b, covering most of the standard normal distribution
    delta_b = 8.0 / (num_b_steps - 1) if num_b_steps > 1 else 8.0 # we calculate the step size for the integration over b

    total_expectation = 0.0 # we initialize the total expectation
    for b in b_values: # we iterate over the values of b
        mu1 = beta * b # we calculate mu1 and mu2 based on beta and b
        mu2 = beta * b
        
        # we calculate the inner expectation for the current b
        inner_expectation = expectation_product_relu(mu1, sigma1, mu2, sigma2, rho, num_steps=num_rho_steps)
        
        # we get the probability density of the current b
        prob_b = norm.pdf(b, loc=0, scale=1)
        
        # we add the weighted inner expectation to the total
        total_expectation += inner_expectation * prob_b * delta_b
        
    return total_expectation # we return the final computed expectation

def main(): # we define the main function
    """
    main function to generate and display the plots.
    """
    # we define a range of correlation values (rho) to plot
    rho_values = jnp.linspace(-0.99, 0.99, 100)
    
    sigma1 = 1.0 # we fix the standard deviation values
    sigma2 = 1.0
    
    # we interpret 'beta' as the mean 'mu'
    # we define several values for the means to see how the plot changes
    beta_values = [0.0]
    
    # --- First Plot: Original Expectation ---
    plt.figure(figsize=(12, 8)) # we create a new figure for the plot
    """
    for beta in beta_values: # we generate a plot for each value of beta (mu)
        mu1 = beta # we assign beta to mu1 and mu2
        mu2 = beta
        
        # we calculate the expectation for each value of rho
        expectation_values = [expectation_product_relu(mu1, sigma1, mu2, sigma2, rho) for rho in tqdm(rho_values, desc=f"Calculating for β={beta}")]
        
        # we plot the results
        plt.plot(np.asarray(rho_values), expectation_values, label=f'β = {beta}')
        
    plt.xlabel("Correlation (ρ)") # we add the labels and title to the plot
    plt.ylabel(r"$\mathbb{E}[\mathrm{ReLU}(X_1)\mathrm{ReLU}(X_2)]$")
    plt.title("Expectation of the product of ReLUs as a function of correlation")
    plt.legend() # we display the legend
    plt.grid(True) # we add a grid for better readability
    plt.savefig(f"../../storage/mmnn_ntk_values/ntk_original_expectation_theory.png")
    plt.show() # we display the plot
    """
    # --- Second Plot: NTK-related Expectation over b ---
     # we create a new figure for the ntk plot

    for beta in beta_values: # we generate a plot for each value of beta
        plt.figure(figsize=(12, 8))
        # we calculate the ntk expectation for each value of rho
        ntk_expectation_values = [calculate_ntk_expectation_over_b(beta, sigma1, sigma2, rho) for rho in tqdm(rho_values, desc=f"Calculating NTK for β={beta}")]

        # we plot the results
        plt.plot(np.asarray(rho_values), ntk_expectation_values, label=f'β = {beta}')
        jnp.savez(f"/home/janis/STG3A/MMNN/data/storage/mmnn_ntk_values/ntk_expectation_values_{beta}_theory.npz", rho_values=rho_values, ntk_expectation_values=ntk_expectation_values)
        plt.xlabel("Correlation (ρ)") # we add the labels and title to the ntk plot
        plt.ylabel(r"$\mathbb{E}_b[\mathbb{E}[\mathrm{ReLU}(X_1)\mathrm{ReLU}(X_2)|b]]$")
        plt.title("NTK-related Expectation as a function of correlation")
        plt.legend() # we display the legend
        plt.grid(True) # we add a grid for better readability
        plt.savefig(f"/home/janis/STG3A/MMNN/data/storage/mmnn_ntk_values/ntk_expectation_over_b_{beta}_theory.png")
        plt.show() # we display the ntk plot

if __name__ == "__main__": # we ensure the script is executed directly
    main() # we call the main function

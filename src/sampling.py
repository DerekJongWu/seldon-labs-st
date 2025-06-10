import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde

def generate_distribution(data_points, method='kde', num_points=1000):
    """
    Generate a probability distribution from sparse data points.
    
    Parameters:
    data_points: array-like
        The observed data points
    method: str
        'kde' for Kernel Density Estimation
        'gaussian' for fitting normal distribution
    num_points: int
        Number of points to generate for the distribution
        
    Returns:
    tuple: (x values, probability density values, fitted distribution object)
    """
    data_points = np.array(data_points)
    
    if method == 'kde':
        # Kernel Density Estimation
        kde = gaussian_kde(data_points)
        x_range = np.linspace(min(data_points), max(data_points), num_points)
        density = kde(x_range)
        return x_range, density, kde
        
    elif method == 'gaussian':
        # Fit normal distribution
        mu, std = stats.norm.fit(data_points)
        x_range = np.linspace(min(data_points), max(data_points), num_points)
        density = stats.norm.pdf(x_range, mu, std)
        return x_range, density, stats.norm(mu, std)
    
def generate_samples(distribution, num_samples=100, method='kde'):
    """
    Generate random samples from the fitted distribution.
    
    Parameters:
    distribution: object
        KDE or scipy.stats distribution object
    num_samples: int
        Number of samples to generate
    method: str
        'kde' or 'gaussian'
        
    Returns:
    array: Random samples from the distribution
    """
    if method == 'kde':
        return distribution.resample(num_samples)[0]
    else:
        return distribution.rvs(size=num_samples)
    
def sample_from_distribution(mean, std_dev, num_samples=1):
    # Generate samples from a normal distribution
    samples = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
    return samples
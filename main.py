import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def trapezoidal_rule(f, a, b, n, *args):
    """Approximate the integral of f from a to b using the trapezoidal rule."""
    x = np.linspace(a, b, n + 1)
    y = f(x, *args)
    h = (b - a) / n
    return h * (np.sum(y) - 0.5 * (y[0] + y[-1]))


def f_odd(x, n):
    """Function for odd student numbers: f(x) = n * sin(πnx)"""
    return n * np.sin(np.pi * n * x)


def a_k(f, k, n, interval, num_points=1000):
    """Compute the Fourier coefficient a_k."""
    a, b = interval
    return (2 / (b - a)) * trapezoidal_rule(lambda x: f(x, n) * np.cos(k * x), a, b, num_points)


def b_k(f, k, n, interval, num_points=1000):
    """Compute the Fourier coefficient b_k."""
    a, b = interval
    return (2 / (b - a)) * trapezoidal_rule(lambda x: f(x, n) * np.sin(k * x), a, b, num_points)


def fourier_series_approximation(f, n, N, x, interval):
    """Compute the Fourier series approximation."""
    a_0 = a_k(f, 0, n, interval) / 2
    series_sum = a_0
    for k in range(1, N + 1):
        series_sum += a_k(f, k, n, interval) * np.cos(k * x) + b_k(f, k, n, interval) * np.sin(k * x)
    return series_sum / 2


def plot_fourier_series_and_function(f, n, N, interval):
    """Plot the Fourier series approximation and the original function on the same plot."""
    x_values = np.linspace(interval[0], interval[1], 1000)
    y_values_approx = [fourier_series_approximation(f, n, N, xi, interval) for xi in x_values]
    y_values_original = f(x_values, n)

    plt.plot(x_values, y_values_original, label='Original Function', color='blue', linestyle='dashed')
    plt.plot(x_values, y_values_approx, label=f'Fourier Series Approximation (N={N})', color='red')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Original Function and Fourier Series Approximation')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_fourier_coefficients(f, n, N, interval):
    """Plot the Fourier coefficients a_k and b_k."""
    a_coeffs = [a_k(f, k, n, interval) for k in range(N + 1)]
    b_coeffs = [b_k(f, k, n, interval) for k in range(1, N + 1)]

    # Plot a_k coefficients
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(N + 1), a_coeffs)
    plt.xlabel('k')
    plt.ylabel('a_k')
    plt.title('Fourier Coefficients a_k')

    # Plot b_k coefficients
    plt.subplot(1, 2, 2)
    plt.bar(range(1, N + 1), b_coeffs)
    plt.xlabel('k')
    plt.ylabel('b_k')
    plt.title('Fourier Coefficients b_k')

    plt.tight_layout()
    plt.show()


def relative_error(f, f_approx, n, interval, num_points=1000):
    """Calculate the relative error between the original function and its Fourier series approximation."""
    x_values = np.linspace(interval[0], interval[1], num_points)
    original_values = f(x_values, n)
    approx_values = np.array([f_approx(xi) for xi in x_values])

    non_zero_mask = np.abs(original_values) > 1e-10  # Маска для ненульових значень
    errors = np.zeros_like(original_values)
    errors[non_zero_mask] = np.abs(original_values[non_zero_mask] - approx_values[non_zero_mask]) / np.abs(
        original_values[non_zero_mask])

    return np.mean(errors)


def absolute_error(f, f_approx, n, interval, num_points=1000):
    """Calculate the absolute error between the original function and its Fourier series approximation."""
    x_values = np.linspace(interval[0], interval[1], num_points)
    original_values = f(x_values, n)
    approx_values = np.array([f_approx(xi) for xi in x_values])
    return np.mean(np.abs(original_values - approx_values))


def calculate_correlation(f, f_approx, n, interval, num_points=1000):
    """Calculate the Pearson correlation coefficient between the original function and its Fourier series approximation."""
    x_values = np.linspace(interval[0], interval[1], num_points)
    original_values = f(x_values, n)
    approx_values = np.array([f_approx(xi) for xi in x_values])
    return pearsonr(original_values, approx_values)[0]


def save_results_to_file(N, a_coeffs, b_coeffs, rel_error, abs_error, corr, filename="fourier_results.txt"):
    """Save the Fourier series results to a file."""
    with open(filename, "w") as file:
        file.write(f"Order of approximation: N = {N}\n")
        file.write("Fourier coefficients a_k:\n")
        for i, a in enumerate(a_coeffs):
            file.write(f"a_{i} = {a}\n")
        file.write("\nFourier coefficients b_k:\n")
        for i, b in enumerate(b_coeffs):
            file.write(f"b_{i + 1} = {b}\n")
        file.write(f"\nRelative error of approximation: {rel_error}\n")
        file.write(f"Absolute error of approximation: {abs_error}\n")
        file.write(f"Correlation between original and approximation: {corr}\n")


def main():
    n = 11  # Replace with your student number
    N = 100  # Order of approximation
    interval = (0, np.pi)
    f = f_odd

    plot_fourier_series_and_function(f, n, N, interval)
    plot_fourier_coefficients(f, n, N, interval)

    approx_func = lambda x: fourier_series_approximation(f, n, N, x, interval)

    rel_error = relative_error(f, approx_func, n, interval)
    abs_error = absolute_error(f, approx_func, n, interval)
    corr = calculate_correlation(f, approx_func, n, interval)

    a_coeffs = [a_k(f, k, n, interval) for k in range(N + 1)]
    b_coeffs = [b_k(f, k, n, interval) for k in range(1, N + 1)]

    save_results_to_file(N, a_coeffs, b_coeffs, rel_error, abs_error, corr)

    print(f"Relative error of approximation: {rel_error}")
    print(f"Absolute error of approximation: {abs_error}")
    print(f"Correlation between original and approximation: {corr}")


def plot_fourier_series_and_function_in_range(f, n, N, interval=(0, 0.5)):
    """Plot the Fourier series approximation and the original function on the interval (0, 0.5)."""

    x_values = np.linspace(interval[0], interval[1], 1000)
    y_values_approx = [fourier_series_approximation(f, n, N, xi, (0, np.pi)) for xi in x_values]
    y_values_original = f(x_values, n)

    plt.plot(x_values, y_values_original, label='Original Function', color='blue', linestyle='dashed')
    plt.plot(x_values, y_values_approx, label=f'Fourier Series Approximation (N={N})', color='red')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Original Function and Fourier Series Approximation on (0, 0.5)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
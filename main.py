import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# core functions
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol(price, S, K, T, r):
    def f(sigma):
        return black_scholes_call(S, K, T, r, sigma) - price
    try:
        if f(1e-6) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-6, 5.0)
    except ValueError:
        return np.nan

# heston functions
def heston_char_func(phi, S, K, T, r, kappa, theta, sigma, rho, v0):
    i = 1j
    a = kappa * theta
    u = -0.5
    b = kappa - rho * sigma
    d = np.sqrt((rho*sigma*i*phi - b)**2 - sigma**2 * (2*u*i*phi - phi**2))
    g = (b - rho*sigma*i*phi + d) / (b - rho*sigma*i*phi - d)
    C = r*i*phi*T + (a/sigma**2)*((b - rho*sigma*i*phi + d)*T - 2*np.log((1-g*np.exp(d*T))/(1-g)))
    D = ((b - rho*sigma*i*phi + d)/sigma**2) * ((1 - np.exp(d*T)) / (1 - g*np.exp(d*T)))
    return np.exp(C + D*v0 + i*phi*np.log(S))

def heston_call_price(S, K, T, r, kappa, theta, sigma, rho, v0):
    def integrand(phi):
        return np.real(np.exp(-1j*phi*np.log(K)) * heston_char_func(phi, S, K, T, r, kappa, theta, sigma, rho, v0) / (1j*phi))
    integral = quad(integrand, 0, 100, limit=200)[0]
    price = S - np.sqrt(S*K)*np.exp(-r*T)/np.pi * integral
    return max(0, min(price, S))

# interactive graph functions
def compute_IV(S, r, kappa, theta, sigma, rho, v0, strikes, maturities):
    IV = np.zeros((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            price = heston_call_price(S, K, T, r, kappa, theta, sigma, rho, v0)
            IV[i, j] = implied_vol(price, S, K, T, r)
    return IV

def interactive_heston_surface():
    S_default = 100
    r_default = 0.01
    kappa_default = 2.0
    theta_default = 0.04
    sigma_default = 0.5
    rho_default = -0.7
    v0_default = 0.04

    strikes = np.linspace(70, 130, 15)       # smaller grid
    maturities = np.linspace(0.1, 2.0, 15)   # smaller grid

    K_grid, T_grid = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.1, bottom=0.35)

    # Initial surface
    IV = compute_IV(S_default, r_default, kappa_default, theta_default,
                    sigma_default, rho_default, v0_default, strikes, maturities)
    surf = ax.plot_surface(K_grid, T_grid, IV, cmap='viridis', edgecolor='k')
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (Years)")
    ax.set_zlabel("Implied Vol")
    ax.set_title("Heston Implied Volatility Surface")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_kappa = plt.axes([0.1, 0.25, 0.8, 0.03], facecolor=axcolor)
    ax_theta = plt.axes([0.1, 0.20, 0.8, 0.03], facecolor=axcolor)
    ax_sigma = plt.axes([0.1, 0.15, 0.8, 0.03], facecolor=axcolor)
    ax_rho = plt.axes([0.1, 0.10, 0.8, 0.03], facecolor=axcolor)
    ax_v0 = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor=axcolor)

    s_kappa = Slider(ax_kappa, 'kappa', 0.1, 5.0, valinit=kappa_default)
    s_theta = Slider(ax_theta, 'theta', 0.01, 0.2, valinit=theta_default)
    s_sigma = Slider(ax_sigma, 'sigma', 0.1, 1.0, valinit=sigma_default)
    s_rho = Slider(ax_rho, 'rho', -0.99, 0.99, valinit=rho_default)
    s_v0 = Slider(ax_v0, 'v0', 0.01, 0.2, valinit=v0_default)

    # Button to update surface
    ax_button = plt.axes([0.45, 0.01, 0.1, 0.04])
    button = Button(ax_button, 'Update Surface', color='lightblue', hovercolor='0.975')

    def update(event):
        ax.clear()
        IV_new = compute_IV(S_default, r_default,
                            s_kappa.val, s_theta.val, s_sigma.val, s_rho.val, s_v0.val,
                            strikes, maturities)
        ax.plot_surface(K_grid, T_grid, IV_new, cmap='viridis', edgecolor='k')
        ax.set_xlabel("Strike")
        ax.set_ylabel("Maturity (Years)")
        ax.set_zlabel("Implied Vol")
        ax.set_title("Heston Implied Volatility Surface")
        fig.canvas.draw_idle()

    button.on_clicked(update)
    plt.show()


if __name__ == "__main__":
    interactive_heston_surface()

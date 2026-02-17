# Implied-Volatility-Surface-Visualizer
**Note:** The Heston pricing functions are adapted from standard open-source implementations. 
My contribution focused on creating the **interactive 3D implied volatility surface** and adding **slider controls** for dynamic exploration.


The project is an implied volatility surface visulizer with interactive buttons and sliders that affects the kappa, theta, sigma, rho, v0 **(more details on these variables later)**

# How it works:
This project takes implied volatiltiy to the next level by putting more focus onto Heston's SDE rather than the black-scholes. The black-scholes is useful, but when trying to find the implied volatility surface it has many downfalls:
1. Firstly, the black-scholes treats volatiltiy as a constant, which obviously isn't the case in the market. In otherwords, we will always be either severely under or overpriced.
2. It asummes a Geometric Brownian Motion. Which means that log-returns should be normally distributed and that we shouldn't have fat tails, jumps or any extreme moves. ***AGAIN, THIS ISN'T THE CASE IN THE MARKETS.*** In reality, asset returns exhibit fat tails, leptokurtosis and skewness.
3. It also fails to capture volatility clustering. This is when large volatility is followed by large volatility and vise-versa rather than being "fully random." This is a little out of the scope of this project, but this could be modeled using ARCH/GARCH.

Now the Heston model assumes stochastic volatility. This means volatility isn't constant and we allow volatility to change over time and be correlated with the stock price. Thus, our surface would be a more accurate dipiction of the market's implied volatility. 

# Code explaination:
Before we continue, I would like to talk about the variables used for the calculations and sliders. 
- kappa – speed of mean reversion of volatility
- theta – long-term variance
- sigma – volatility of volatility
- rho – correlation between asset and volatility
- v0 – initial variance

I first created the function black_scholes_call(S, K, T, r, sigma):
~~~ python
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
~~~
This function takes in the paramaters: 
- S → current stock price,
- K → strike priceT → time to maturity (in years),
- r → risk-free interest rate,
- sigma → volatility

This function just calculates a theoretical European option call price.

Next, we calculated the implied volatility. 
~~~ python
def implied_vol(price, S, K, T, r):
    def f(sigma):
        return black_scholes_call(S, K, T, r, sigma) - price
    try:
        if f(1e-6) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-6, 5.0)
    except ValueError:
        return np.nan
~~~
This uses the previous black-scholes function and checks if f(1e-6) and f(5.0) have opposite signs (a requirement for Brent’s method).

Then we compute the characteristic function of the Heston model for Fourier-based option pricing. This allows pricing options under stochastic volatility and reproduces real-world smile/skew effects.
~~~ python
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
~~~
characteristic function -> a complex function that completely determines the probability distribution of a random variable.

Lastly, we compute a European call price using the Heston model by numerical integration of the characteristic function. This implements a more realistic model than Black-Scholes, accounting for stochastic volatility and correlation with asset returns.
~~~ python
def heston_call_price(S, K, T, r, kappa, theta, sigma, rho, v0):
    def integrand(phi):
        return np.real(np.exp(-1j*phi*np.log(K)) * heston_char_func(phi, S, K, T, r, kappa, theta, sigma, rho, v0) / (1j*phi))
    integral = quad(integrand, 0, 100, limit=200)[0]
    price = S - np.sqrt(S*K)*np.exp(-r*T)/np.pi * integral
    return max(0, min(price, S))
~~~

The rest of the code are ways to visualize these primary functions and interact with them using sliders.

# Issues while writing the code:
The interactive implied volatility surface is computationally intensive because each update requires:
- Pricing options using the Heston model (numerical integration of the characteristic function).
- Computing the implied volatility via root-finding for each strike-maturity combination.

As a result, live slider updates (continuous dragging) can be slow or unresponsive in Matplotlib.

Mitigation:
- The “Update Surface” button triggers recalculation only once after adjusting sliders, keeping the interface responsive.
- Grid size and numerical integration parameters are chosen to balance performance and visual fidelity.

Future improvements could include caching or vectorized computation to enable near-real-time slider updates.

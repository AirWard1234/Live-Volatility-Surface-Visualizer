# Implied-Volatility-Surface-Visualizer
Note, the project changed from live to implied, so the content of the repository will have some slight changes.

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


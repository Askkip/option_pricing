import numpy as np
from scipy.stats import norm

def black_scholes(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return call_price, put_price

# Exemple d'utilisation
S = 116.75  # Prix de l'actif sous-jacent
K = 150  # Prix d'exercice de l'option
r = 0.05  # Taux d'intérêt sans risque
sigma = 0.3173 # Volatilité du sous-jacent
T = 1 # Temps jusqu'à l'expiration de l'option en année

call_price, put_price = black_scholes(S, K, r, sigma, T)
print("Prix de l'option d'achat :", call_price)
print("Prix de l'option de vente :", put_price)

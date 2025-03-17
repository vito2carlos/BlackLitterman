# BlackLitterman

The BlackLitterman model uses a Bayesian approach for asset allocation. It combines **prior**(initial) expected returns (like the CAPM) with investor **views** on assets to produce a **posterior**(updated) estimate of expected returns.
This is an implementation of the model for the Moroccan financial market, with a GUI to input **views**.

Essentially, Black-Litterman treats the vector of expected returns itself as a quantity to be estimated.

The Black-Litterman formula:



<p>
  $$ E(R) = [(\tau \Sigma)^{-1} + P^T \Omega^{-1} P]^{-1}[(\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q] $$
</p>

Parameters:

- E(R):  vector of expected returns, where N is the number of assets.

- Q:  vector of views.

- P:  picking matrix, mapping views to the universe of assets. It indicates which view corresponds to which asset(s).

- Ω (Omega):  uncertainty matrix of views.

- Π (Pi):  vector of prior expected returns.

- Σ (Sigma):  covariance matrix of asset returns.

- τ (Tau): Scalar tuning constant.
 
 ## Environment

 - Windows 11

Developed and tested in this environment.

## Run

To run the application, please run the lines below.

```shell
$ git clone https://github.com/vito2carlos/BlackLitterman.git
$ cd BlackLitterman
$ python3 app.py

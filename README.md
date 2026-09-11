# VaR and Backtesting

## UC Berkeley MFE — Risk Management

This project analyzes **Value at Risk (VaR)** estimation and backtesting for a
$10 million equity portfolio as part of the UC Berkeley Master of Financial
Engineering Risk Management course.

The project compares parametric and non-parametric approaches to market-risk
measurement and studies how estimation methodology and rolling-window length
affect VaR forecasts and their reliability.

---

## Project Overview

On August 31, 2022, a $10 million equity portfolio is constructed containing
**GameStop (GME)** and nine other companies of similar size.

Each stock has at least a 5% portfolio weight, and the portfolio is held without
rebalancing.

The analysis forecasts **95% one-day Value at Risk** from September 1, 2022
through August 1, 2026 and evaluates the performance of alternative VaR
methodologies.

---

## Objectives

The project addresses eight main questions.

### 1. Portfolio Construction

Construct a $10 million portfolio containing:

- GameStop (`GME`)
- nine other stocks from companies of similar size
- a minimum 5% weight in every stock

The selected portfolio is held without rebalancing.

### 2. Parametric VaR

Estimate the **95% one-day VaR** using a rolling 100-day
variance-covariance matrix under the assumption that stock returns follow a
multivariate normal distribution.

A one-factor model is also estimated and the corresponding beta dynamics are
analyzed.

### 3. Historical Simulation VaR

Estimate the 95% one-day VaR using **historical simulation** with a rolling
100-day window.

Unlike parametric VaR, historical simulation does not impose a normal
distribution on portfolio returns.

### 4. Marginal VaR

Estimate **Marginal VaR** for:

- GME
- one additional stock in the portfolio

Marginal VaR measures how total portfolio risk changes when exposure to an
individual asset changes.

### 5. Bootstrap Confidence Interval

Construct **95% bootstrap confidence intervals** around the historical VaR
forecasts.

Bootstrap resampling is used to quantify estimation uncertainty in the VaR
series.

### 6. 500-Day Rolling Window

Repeat the historical simulation, marginal VaR, and bootstrap analysis using a
larger **500-day rolling window**.

This allows comparison of short-window and long-window VaR estimates.

### 7. VaR Backtesting

Evaluate the VaR forecasts using the preceding 500 trading days.

A VaR exception occurs when the realized portfolio loss exceeds the predicted
VaR:

\[
\text{Loss}_t > VaR_t
\]

The backtesting exercise compares:

- Parametric VaR
- Historical Simulation VaR — 100-day window
- Historical Simulation VaR — 500-day window

The number and frequency of exceptions are used to assess model reliability.

### 8. Discussion

The final analysis considers:

- spikes in the VaR series
- the effect of portfolio composition on risk
- reliability of the different VaR approaches
- differences between parametric and historical simulation methods
- differences between 100-day and 500-day estimation windows
- possible improvements to the VaR forecasting framework

---

# Methodology

## Portfolio Returns

Let

\[
w =
\begin{bmatrix}
w_1 & w_2 & \cdots & w_N
\end{bmatrix}^{\top}
\]

denote portfolio weights and

\[
r_t =
\begin{bmatrix}
r_{1,t} & r_{2,t} & \cdots & r_{N,t}
\end{bmatrix}^{\top}
\]

denote asset returns.

The portfolio return is

\[
r_{p,t} = w_t^{\top}r_t.
\]

Because the portfolio is held without rebalancing, portfolio weights can evolve
through time as asset prices change.

---

## Parametric VaR

Under the multivariate-normal assumption, portfolio volatility is

\[
\sigma_{p,t}
=
\sqrt{w_t^{\top}\Sigma_t w_t},
\]

where \(\Sigma_t\) is the rolling covariance matrix of stock returns.

The 95% one-day VaR is based on the 95% standard-normal quantile:

\[
VaR_{t,0.95}
=
z_{0.95}\sigma_{p,t}.
\]

The analysis uses a **100-day rolling estimation window**.

---

## One-Factor Model

A one-factor model is used to describe common variation in portfolio returns.

A general specification is

\[
r_{i,t}
=
\alpha_i
+
\beta_i r_{m,t}
+
\epsilon_{i,t}.
\]

Rolling estimates of factor exposure are used to study changes in systematic
risk through time.

---

## Historical Simulation VaR

Historical simulation estimates VaR directly from the empirical distribution
of past portfolio returns.

For confidence level \(c=0.95\),

\[
VaR_{0.95}
=
-
Q_{0.05}(r_p),
\]

where \(Q_{0.05}\) is the 5th percentile of historical portfolio returns.

Two rolling windows are studied:

- 100 trading days
- 500 trading days

---

## Marginal VaR

Marginal VaR measures the sensitivity of portfolio VaR to the exposure of an
individual asset:

\[
MVaR_i
=
\frac{\partial VaR_p}{\partial w_i}.
\]

The project evaluates Marginal VaR for GME and another portfolio stock.

---

## Bootstrap Confidence Intervals

Bootstrap sampling is used to characterize uncertainty around historical VaR.

For each rolling estimation window:

1. returns are sampled with replacement;
2. VaR is recalculated for each bootstrap sample;
3. the empirical distribution of bootstrap VaRs is constructed;
4. the 2.5th and 97.5th percentiles form the 95% confidence interval.

---

# Repository Structure

```text
VaR and Backtesting/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
│
├── ass/
│
├── data/
│   ├── raw/
│   │   └── stock_prices.csv
│   │
│   └── processed/
│       ├── stock_returns.csv
│       ├── Q02_parametric_var.csv
│       ├── Q02_one_factor_model.csv
│       ├── Q03_historical_var_100.csv
│       ├── Q04_marginal_var_100.csv
│       ├── Q05_bootstrap_ci_100.csv
│       ├── Q06_historical_var_500.csv
│       ├── Q06_marginal_var_500.csv
│       └── Q06_bootstrap_ci_500.csv
│
├── notebooks/
│   ├── 01_VaR_Analysis.ipynb
│   ├── 02_Backtesting_Q7_final.ipynb
│   └── riskmanagement_NLP_text.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│
├── report/
│
└── src/

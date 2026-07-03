# BTC-RL-Agent: Reinforcement Learning for Crypto Market Modeling

**ORIE 5570 — Reinforcement Learning with Operations Research Applications**  
Cornell University · Spring 2026

Four-part research project applying classical and deep RL to Bitcoin (BTCB-USD) rational investor modeling. Covers tabular Q-learning through policy gradients and Bayesian bandits, with formal MDP derivations, regime-conditional backtesting, and bootstrap-validated risk metrics.

---

## Algorithm Catalog

| Part | Algorithm | Setting | Key Result |
|------|-----------|---------|------------|
| 1–2 | **TD Q-Learning** (tabular) | Discretized MDP, ε-greedy | Steady-state Sharpe vs. B&H baseline |
| 1–2 | **DQN** (Double Q-Network) | Continuous state, replay buffer, target net | Outperforms tabular in rapid-rise regime |
| 3 | **REINFORCE + Baseline** | Policy gradient, 3 trader types | Manipulator: −0.06 total return vs. −0.46 B&H |
| 3 | **Double DQN** | 3-layer MLP, 800 episodes | Sharpe −3.94 vs. −3.20 B&H |
| 4 | **Thompson Sampling** | Beta-Bernoulli conjugate, 5 crypto arms | Final regret 9.19 ± 0.78 (T=500, N=100) |
| 4 | **Bayes-UCB** | Posterior quantile exploration | Final regret 9.13 ± 0.85 — lowest across algorithms |
| 4 | **UCB1** | Hoeffding-bound confidence intervals | Final regret 9.45 ± 0.85 |

---

## Mathematical Framework

### MDP Formulation

The trading problem is cast as a finite-horizon MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- **State** $s_t \in \mathbb{R}^{13}$: 12 engineered market features + portfolio crypto fraction $f_t \in [0,1]$
- **Action** $a_t \in \{0\ (\text{hold}),\ 1\ (\text{buy}),\ 2\ (\text{sell})\}$
- **Reward** $r_t = \log(V_{t+1}/V_t) - c \cdot \mathbf{1}[\text{trade}]$, where $c = 0.001$ (10 bps transaction cost)
- **Discount** $\gamma = 0.99$

The Bellman optimality equation driving all value-based methods:

$$Q^*(s,a) = \mathbb{E}\left[r_t + \gamma \max_{a'} Q^*(s_{t+1}, a') \;\Big|\; s_t = s,\, a_t = a\right]$$

### Feature Engineering

12 signals extracted from hourly OHLCV (17,080 observations, Mar 2024 – Feb 2026):

| Feature | Formula | Economic Rationale |
|---------|---------|-------------------|
| `ret_1h`, `ret_4h`, `ret_24h`, `ret_168h` | $r_h = (P_t - P_{t-h})/P_{t-h}$ | Multi-horizon momentum |
| `vol_24h` | $\sigma_{24} = \text{std}(\log P_t/P_{t-1},\ 24h)$ | Realized volatility |
| `rsi_14` | $100 - 100/(1 + \text{AvgGain}/\text{AvgLoss})$ | Mean-reversion signal |
| `price_vs_ma20` | $(P_t - \mu_{20})/\sigma_{20}$ | Bollinger z-score |
| `price_vs_ma50` | $(P_t - \mu_{50})/\mu_{50}$ | Trend deviation |
| `hour_sin`, `hour_cos` | $\sin/\cos(2\pi h/24)$ | Intraday seasonality |
| `hl_range` | $(H_t - L_t)/P_t$ | Intraday volatility proxy |

### Market Regime Detection

Regimes labeled on 24h trailing return $r_{24}$:

$$\text{regime}_t = \begin{cases} \text{decline} & r_{24} < -5\% \\ \text{rise} & r_{24} > +5\% \\ \text{steady} & \text{otherwise} \end{cases}$$

---

## Part 3: REINFORCE — Heterogeneous Trader Types

Three behaviorally distinct reward functions model different market participants:

| Trader Type | Reward Function | Behavior |
|-------------|----------------|----------|
| **Arbitrageur** | Log P&L scaled by volatility | Exploits mispricings, high turnover |
| **Manipulator** | Directional P&L + momentum bonus | Trend-following with position amplification |
| **Herder** | Consensus-tracking reward | Follows crowd signal, low conviction |

Policy gradient update (REINFORCE with learned baseline $b_\phi(s_t)$):

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)\, (G_t - b_\phi(s_t))\right]$$

**Backtest results on held-out test set (3,416 hours):**

| Strategy | Total Return | Sharpe | Max Drawdown | Final Portfolio |
|----------|-------------|--------|--------------|-----------------|
| REINFORCE-Manipulator | −6.01% | −3.94 | −6.23% | $9,399 |
| REINFORCE-Herder | −17.24% | **−1.55** | −26.32% | $8,276 |
| REINFORCE-Arbitrageur | −53.07% | −5.42 | −56.23% | $4,693 |
| Double DQN | −49.44% | −5.30 | −53.18% | $5,056 |
| **Buy & Hold** | −46.21% | −3.20 | −49.99% | $5,379 |

SHAP analysis identifies RSI-14, 24h return, and price-vs-MA50 as dominant features across all trader types.

---

## Part 4: Bayesian Multi-Armed Bandit — Crypto Portfolio Selection

**5 arms:** BTC (θ=0.5153), ETH (θ=0.5143), SOL (θ=0.5038), **BNB (θ\*=0.5189)**, USDC (θ=0.5053)  
**Setup:** T=500 timesteps, 100 bootstrap replications, Beta(3,3) prior

**Thompson Sampling** information ratio (Russo & Van Roy 2014):

$$\Gamma_t = \frac{\left(\mathbb{E}_\pi[\Delta_t(A_t)]\right)^2}{I_t(\theta^*;\ A_t)}$$

Empirical $\bar{\Gamma}_T = 0.149$ against theoretical upper bound of $0.5$, confirming efficient exploration.

**Final cumulative regret at T=500 (95% CI):**

| Algorithm | Mean Regret | ±1.96×SE | 95% CI | Best/100 |
|-----------|-------------|----------|--------|----------|
| UCB1 | 9.452 | 0.852 | [8.60, 10.30] | 16/100 |
| **Bayes-UCB** | **9.125** | 0.848 | [8.28, 9.97] | 18/100 |
| Thompson Sampling | 9.190 | 0.781 | [8.41, 9.97] | 21/100 |
| Greedy | 9.150 | 1.637 | [7.51, 10.79] | — |

Bayes-UCB achieves lowest mean regret; Thompson Sampling achieves lowest variance — consistent with $O(\sqrt{KT \log T})$ theoretical guarantees.

---

## Repository Structure

```
btc-rl-agent/
├── rl_crypto_assignment.ipynb       # Parts 1–2: Tabular Q-Learning + DQN
├── rl_crypto_part3.ipynb            # Part 3: REINFORCE + Baseline, 3 trader types
├── rl_crypto_part3_executed.ipynb   # Part 3 with full outputs
├── rl_crypto_part4.ipynb            # Part 4: Bayesian Bandits (UCB1, Bayes-UCB, TS)
├── rl_crypto_part4_executed.ipynb   # Part 4 with full outputs
├── BTCB-USD_DataHr.csv              # Hourly OHLCV, Mar 2024 – Feb 2026 (17,080 rows)
│
├── reports/
│   ├── RL_Crypto_Report_Final.pdf   # Final consolidated research report
│   ├── RL_Crypto_Report.pdf         # Intermediate report
│   ├── RL_Crypto_Part3_Report.pdf   # Part 3 writeup
│   ├── RL_Crypto_Part3_Report.tex   # Part 3 LaTeX source
│   ├── RL_Crypto_Part4_Report.pdf   # Part 4 writeup
│   ├── RL_Crypto_Part4_Report.tex   # Part 4 LaTeX source
│   ├── RL_Reinforce.pdf             # REINFORCE derivation writeup
│   └── ORIE 5570 Assignment 1 Literature Review.pdf
│
└── slides/
    ├── RL Final Presentation.pdf    # Final course presentation
    ├── RL_Crypto_Slides.pdf         # Compiled slide deck
    └── RL_Crypto_Slides.tex         # LaTeX Beamer source
```

---

## Stack

- **Python 3.11** · PyTorch 2.9 · NumPy · Pandas · Gymnasium
- **Stable-Baselines3** (PPO/SAC reference)
- **SHAP** for policy interpretability (KernelExplainer)
- **SciPy** for bootstrap CI and Beta conjugate posteriors
- **Matplotlib** for all figures

---

## Data

`BTCB-USD_DataHr.csv` — Bitcoin BEP2 hourly OHLCV sourced from Yahoo Finance.  
17,080 observations · March 2024 – February 2026 · 80/20 train/test split.

---

*ORIE 5570 · Cornell University · Ishan Patwardhan (ip259)*

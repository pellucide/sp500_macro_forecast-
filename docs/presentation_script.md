# Project Presentation Script — Can Economic Data Predict the Stock Market?

Target: ~5 minutes. Walkthrough while pointing at poster sections.

---

## [0:00] Intro
"Hi, I'm Jagat Brahma. For my CS 229 project, I tried something ambitious — maybe a little foolish — I tried
to use economic data to predict the stock market. I did not get rich. But I learned something interesting."

## [0:15] The Pipeline — *point at pipeline diagram*
"Here's how it works. I pull 134 economic indicators from the Fed — jobs, inflation, housing, interest rates.
That's 134 dimensions of data. But most of them move together — when the economy is strong, employment rises,
housing starts rise, retail sales rise. So I use PCA, or principal component analysis, to find the few
independent directions that capture most of the variation. It compresses all that noise into maybe 10
meaningful factors. The idea is to turn a firehose of noisy data into something a model can actually use."

## [0:45] What We Built / Data — *point to left column*
"I'm predicting whether the S&P 500 goes up or down — next month, and next quarter. I use a walk-forward
backtest: start with 10 years of training data, predict the next month, then expand the window and repeat.
Every prediction uses only past data. No cheating, no lookahead bias. This simulates real-time trading
from 2000 to 2026 — 26 years of honest evaluation."

## [1:10] Features — *point to Features section*
"Raw economic data is noisy and highly correlated. So first, I run a univariate regression for each indicator
against returns and compute its t-statistic. Within each category, I keep only the indicators whose t-stat
exceeds a threshold — say, the 3-month T-bill yield survives but 9 others don't. This prevents one category
from flooding the model. Then I scale each survivor by its regression slope — if unemployment has been a
strong predictor, it gets amplified. Finally, PCA compresses everything into orthogonal factors, each
one an independent economic signal. I also multiply by recent volatility so the model knows when to be
cautious — calm vs turbulent markets behave differently."

## [1:45] Models — *point to Models table*
"I tested seven models, from simple to complex. Linear regression gives you a weighted sum of the factors.
Elastic Net adds L1 and L2 regularization — it penalizes the model for using too many features, which
prevents overfitting when you have more predictors than data points. Tree-based models like XGBoost and
Random Forest can learn non-linear interactions — maybe low unemployment matters more when rates are also
low. An ensemble averages linear and XGBoost predictions, which smooths out their individual biases."

## [2:15] Results — *point to Results table*
"For monthly predictions, nothing beats always-long. The market goes up 62.5% of months — you can't beat
that with any amount of economic data. And R-squared out of sample is negative for every model. That means
you'd be better off just predicting the historical average. So monthly forecasting is basically a fool's
errand."

"But look at the quarterly horizon. CatBoost hits 74.5% accuracy, beating the 66.6% baseline by nearly
8 points. The ensemble gets a Sharpe ratio of 0.87 — that's risk-adjusted return, and it beats buy-and-hold
at 0.68. These use overlapping returns — consecutive predictions share data — so I also tested non-overlapping
evaluation with independent quarters. Random Forest gets 72.3%. I ran a Diebold-Mariano test for statistical
significance, and the results hold. The signal is real, not an artifact."

## [2:55] Key Takeaways — *point to Key Takeaways box*
"Three takeaways.
One: monthly is hopeless — the signal-to-noise ratio is just too low at that horizon.
Two: at the quarterly horizon, there's real directional signal, confirmed in both overlapping and
non-overlapping evaluations.
Three: models get direction right but magnitude wrong. All R-squared values are negative. The model knows
which way the market is heading, but not how far. The signal exists, but it's modest."

## [3:25] Discussion / What Didn't Work — *point to right column*
"Not everything worked. Negative R-squared across the board means a simple historical average predicts
magnitudes better than any of our models. Extreme leverage at 25x destroys everything — the directional
accuracy isn't high enough to sustain that kind of amplification. With asymmetric leverage at 2.5x long
and 0.25x short, max drawdown drops from 47% to around 10%. Transaction costs eat 1-2% of returns.
And as I said, monthly forecasting is hopeless."

## [3:55] Future Work — *point to Future Work*
"For a production version, I'd model the lag in economic data releases — FRED reports come weeks after
month-end, so we're essentially using stale news that the market already priced in. And I'd replace my
simple volatility proxy with a hidden Markov model for better regime detection — instead of just measuring
volatility, it would learn distinct market states like bull, bear, and calm."

## [4:15] Close
"Code is on GitHub. Maybe one day this actually makes money. Until then, it's a solid party trick.
Thanks for listening."

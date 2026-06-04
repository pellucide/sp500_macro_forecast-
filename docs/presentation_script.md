# Project Presentation Script — Can Economic Data Predict the Stock Market?

Target: ~4 minutes. Walkthrough while pointing at poster sections.

---

## [0:00] Intro
"Hi, I'm Jagat Brahma. For my CS 229 project, I tried something ambitious — maybe a little foolish — I tried
to use economic data to predict the stock market. I did not get rich. But I learned something interesting."

## [0:15] The Pipeline — *point at pipeline diagram*
"Here's how it works. I pull 134 economic indicators from the Fed — jobs, inflation, housing, interest rates.
Basically CNBC's diet. Then I run them through a pipeline: screen for signal, scale by predictive strength,
compress into 10 PCA factors, and adjust for market volatility. The idea is to turn a firehose of noisy data
into something a model can actually use."

## [0:45] What We Built / Data — *point to left column*
"I'm predicting whether the S&P 500 goes up or down — next month, and next quarter. I simulated 26 years of
real-time trading. Every month, the model only gets to see the past. No cheating, no lookahead bias. If this
were a person, they'd be very disciplined and slightly disappointed."

## [1:10] Features — *point to Features section*
"Raw economic data is noisy and highly correlated. So first, I group indicators by category and keep only the
ones that actually show signal. This prevents, say, 20 interest rate indicators from drowning everything else
out. Then I scale each survivor by how well it has historically predicted returns. If unemployment has been a
strong predictor, it gets amplified. If housing starts have been useless, it gets ignored. Finally, PCA compresses
everything into 10 economic factors — a labor factor, a rates factor, an inflation factor. I also multiply by
recent volatility so the model knows when to be cautious."

## [1:50] Models — *point to Models table*
"I tested seven models: linear regression, elastic net, XGBoost, CatBoost, random forest, a neural net, and an
ensemble that averages linear + XGBoost. The baselines are always-long — like Warren Buffett, the 'buy and hold'
special — and momentum, which assumes whatever happened this month will happen again."

## [2:15] Results — *point to Results table*
"Here's where it gets interesting. For monthly predictions, nothing beats always-long. The market goes up 62.5%
of months — you can't beat that with any amount of economic data. If your model says 'down' 40% of the time, you're
wrong 40% of the time. So monthly forecasting is basically a fool's errand."

"But look at the quarterly horizon. CatBoost hits 74.5% accuracy, beating the 66.6% baseline by nearly 8 points.
The ensemble gets 70.3% with the best risk-adjusted return. So there is signal — you just have to be patient."

"Now, these results use overlapping returns — consecutive predictions share two thirds of the same data, which
can inflate the numbers. So I also tested with independent quarterly observations using a non-overlapping
evaluation. Random Forest gets 72.3%, CatBoost gets 67.7% — both beat the baseline. The signal is real."

## [2:55] Key Takeaways — *point to Key Takeaways box*
"Five takeaways.
One: It was really hard to predict next month.
Two: at the quarterly horizon, there's real — though modest — signal. Both overlapping and non-overlapping
evaluations confirm it.
Three: Random Forest gives the cleanest non-overlapping results at 72.3%.
Four: models get direction right but magnitude wrong.
And five: the signal exists at the quarterly horizon, not the monthly one."

## [3:20] Discussion / What Didn't Work — *point to right column*
"Not everything worked. All models have negative R-squared out of sample for magnitude prediction — meaning a
simple average is better at predicting how much the market will move. Leverage beyond 5x? Destroys everything.
With a leverage of 2.5x, this model works best. Transaction costs eat 1-2% of returns. And as I said, monthly
forecasting is hopeless. Some lessons are expensive. These were free."

## [3:45] Future Work — *point to Future Work*
"For a production version, I'd model the lag in economic data releases — FRED reports come weeks after month-end,
so we're essentially trading on stale news. And I'd replace my simple volatility proxy with a hidden Markov model
for better regime detection."

## [4:00] Close
"Code is on GitHub. Maybe one day this actually makes money. Until then, it's a solid party trick. Thanks for listening."

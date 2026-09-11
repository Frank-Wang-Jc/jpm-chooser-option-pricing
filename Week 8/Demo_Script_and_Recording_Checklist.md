# 5–10 Minute Demonstration Script

## 0:00–0:45 — Project objective

Introduce the JPM simple chooser option and explain that the holder chooses call or put
at T1. State the pipeline: real JPM/VIX/Treasury data, BSM baseline, two ML approaches,
validation, stress testing and the final tool.

## 0:45–2:00 — Data and baseline

Show the cached/live status and the default inputs. Explain K=$150, T1=0.5 and T2=1.0.
Point out the BSM price and Greeks. Mention that chooser transactions are OTC and no
public historical chooser transaction series was available, so the project never treats
JPM stock close as an option price.

## 2:00–3:30 — Dual pricing

Compare the BSM price, selected Random Forest volatility-plus-BSM price and direct-price
benchmark. State the final metrics shown in the application. Describe the 90% interval as
an out-of-fold proxy-error band rather than a market bid/ask spread.

## 3:30–5:00 — Sensitivity and stress tests

Move volatility and interest-rate inputs. Show the volatility curve, rate curve and the
required 50% volatility-spike and 2% rate-hike scenarios. Explain why the chooser value
normally rises with volatility.

## 5:00–6:30 — Performance

Open the model-performance tab. Show MAE, RMSE and R² for BSM, Approach 1 and Approach 2,
then show the test-period trend chart. Emphasize the chronological 70/15/15 split.

## 6:30–7:30 — Data refresh and limitations

Press Refresh public data and show the source dates and common feature date. Explain
that all original features refresh together, automatically every 15 minutes while the
page remains open. A provider timeout retains cached data with an explicit status.
Live updates use completed daily observations, not intraday ticks. Finish with the
proxy target, out-of-training-range warnings and no investment-advice claim.

## Recording checklist

- Use 1080p and enlarge browser text before recording.
- Run the packaged app locally using its README; a public deployment is not required.
- Do one practice run under eight minutes.
- Verify that source dates and the data mode are visible.
- Do not call the proxy target an actual chooser market price.
- Submit the MP4 with the GitHub repository, final report PDF and presentation deck.

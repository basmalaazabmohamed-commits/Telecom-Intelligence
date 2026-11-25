
# 27. Final Conclusions

## Predictive Performance

The final selected model is **Tuned XGBoost**.

On the independent test dataset of 15,000
customers, the model achieved:

- ROC-AUC: 0.6937
- Accuracy: 59.15%
- F1-score: 0.6887
- Recall: 91.18%
- Precision: 55.33%
- Brier Score: 0.2207

The bootstrap 95% confidence interval for ROC-AUC is
0.6851
to 0.7017.

## Business Targeting Performance

When targeting the top 20% of customers ranked by
predicted churn risk, the system achieves a lift of
1.50 relative to random targeting.

The revenue-aware decision framework selects
3,000 customers under the
specified campaign assumptions.

## Financial Impact Simulation

Under the base-case assumptions, the campaign is
estimated to retain revenue of
176,251.56
dataset currency units.

The estimated net contribution is
45,750.93
currency units, with a simulated campaign ROI of
76.25%.

These are scenario-based estimates, not experimentally
verified financial outcomes.

## System Reliability

The project includes a reusable prediction engine,
model persistence, automated system testing, and
simulated data-drift monitoring.

Automated system tests passed:
True.

## Recommendation

Company A should evaluate the proposed system through
a controlled retention pilot before operational rollout.

The pilot should verify incremental retention effects,
financial contribution, probability calibration and
customer experience.

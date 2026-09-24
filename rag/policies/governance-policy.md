# Model Governance Policy

Every production model must have a governance profile containing owner, version, dataset, training date, metrics, drift, bias, explainability, security, cost and an overall risk score.

Models with an overall risk score of 30 or below are automatically approved. Models scoring above 30 require human review by a model risk officer. Scores above 60 are escalated and require sign-off from the head of model risk.

Any single risk component of 75 or above forces human review regardless of the overall score.

Re-evaluation is mandatory after every retraining and at least weekly for models in production.

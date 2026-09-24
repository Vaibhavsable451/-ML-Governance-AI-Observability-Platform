# Model Risk Policy

Data drift is measured with the Population Stability Index (PSI). A PSI below 0.10 is stable, between 0.10 and 0.25 is a warning, and above 0.25 is critical and requires investigation.

Performance degradation is measured as the drop in ROC AUC against the validation baseline. A drop of 0.05 or more is critical concept drift and requires retraining or rollback.

Fairness is assessed with demographic parity difference, equal opportunity difference and the four-fifths disparate impact rule. A disparate impact ratio below 0.80 is a policy violation.

Models whose protected attribute ranks among the top three drivers of predictions must be reviewed for indirect discrimination.

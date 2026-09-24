# Security Policy

All prompts must pass prompt-injection and jailbreak screening before reaching a model. Requests that attempt to ignore instructions, reveal system prompts or override safety controls are blocked and logged as security incidents.

Requests must not attempt to disable guardrails, monitoring or logging, or bypass the approval process.

Security incidents feed directly into the security component of the model risk score.

"""Spatially-honest GPS-jitter and noisy-label diagnostics for satellite-based
poverty mapping in Ethiopia.

Rewritten from a prototype whose "buffer beats naive point extraction"
result was a hardcoded parameter, not a simulated consequence of geography,
and whose "noisy test data severely underestimates performance" headline
didn't match its own numbers. This version replaces the hardcoded recovery
fractions with an actual spatially-autocorrelated wealth field: GPS jitter
really displaces the sampled point, and the buffer really is guaranteed to
contain the true location, so whatever recovery advantage buffer pooling
gets is a consequence of that geometry, not a knob.
"""

__version__ = "1.0.0"

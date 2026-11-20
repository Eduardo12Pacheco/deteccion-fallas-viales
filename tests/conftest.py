import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=100,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
if os.environ.get("CI"):
    settings.load_profile("ci")

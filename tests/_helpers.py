import os
from pathlib import Path


def docker_available() -> bool:
    """True only if running locally with a docker socket containing live containers.

    Returns False in any CI environment (GitHub Actions / Forgejo Actions sets CI=true)
    even if a docker socket exists, since CI runners typically have an empty daemon.
    """
    if os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true":
        return False
    return Path("/var/run/docker.sock").exists()

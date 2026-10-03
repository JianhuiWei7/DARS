class ShortTerm:
    """A reward term that returns the wrong number of rewards (for failure-path tests)."""

    def __call__(self, traj, annotation, potentials, rewards):
        return rewards[:-1]

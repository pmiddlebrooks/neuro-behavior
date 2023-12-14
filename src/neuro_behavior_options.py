class neuro_behavior_options:
    def __init__(self):
        # Default options for ridge regression analyses

        # Sampling frequency for various collected data
        self.fsBhv = 60
        self.fsSpike = 30000
        self.fsLfp = 1250

        self.frameSize = 0.1  # seconds per time step (frame)
        self.windowSize = 2  # seconds: use a 2-second window (one second before and after behavior onset)
        self.framesPerTrial = int(self.windowSize / self.frameSize)

        # To capture a window centered on the start of behavior, shift the time bins by half their width.
        self.shiftAlignFactor = 0  # -.5 will make windows centered on regressor events instead of flanking them before/after

        # How many frames do we want to regress before and after events?
        self.sPostTime = int(1 / self.frameSize)  # frames: follow stim events for 1000ms for sPostStim (used for eventType 2)
        self.mPreTime = int(1 / self.frameSize)  # frames: precede motor events by 1000 ms to capture preparatory activity (used for eventType 3)
        self.mPostTime = int(1 / self.frameSize) - 1  # frames: follow motor events for 1000 ms for mPostStim (used for eventType 3). Remove one frame since behavior starts at time zero

        self.folds = 10  # number of folds for cross-validation

        # Use 30 minutes of data
        self.collectStart = 30 * 60  # seconds: when to start collecting the data
        self.collectFor = 30 * 60  # seconds: how long to collect data

        # Balance the data (take nTrial of each behavior)
        self.nTrialBalance = 200

        # Only count (use) a behavior if it...
        self.minActTime = 0.099999  # is at least minActTime long (in seconds)
        self.minNoRepeatTime = 0.99999  # can't have occurred this much time in the past (in seconds)
        self.minBhvNum = 20  # must have occurred at least this many times to analyze

        # For one-back (previous behavior) analyses
        self.minOneBackNum = 40
        self.nOneBackKeep = 6

        # Neural activity options
        self.removeSome = True
        self.minFiringRate = 0.1
        self.maxFiringRate = 40


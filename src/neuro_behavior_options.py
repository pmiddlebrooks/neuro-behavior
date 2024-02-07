def neuro_behavior_options():
    opts = {
        'fsBhv': 60,
        'fsSpike': 30000,
        'fsLfp': 1250,
       
        'frameSize': 0.1,  # seconds per time step (frame)
        'windowSize': 2,  # seconds: use a 2-second window (one second before and after behavior onset)
        'framesPerTrial': int(2 / 0.1),  # Automatically calculate this from windowSize and frameSize
       
        'shiftAlignFactor': 0,  # -.5 will make windows centered on regressor events instead of flanking them before/after
       
        'sPostTime': int(1 / 0.1),  # frames: follow stim events for 1000ms for sPostStim (used for eventType 2)
        'mPreTime': int(1 / 0.1),  # frames: precede motor events by 1000 ms to capture preparatory activity (used for eventType 3)
        'mPostTime': int(1 / 0.1) - 1,  # frames: follow motor events for 1000 ms for mPostStim (used for eventType 3), adjust for behavior start time
       
        'folds': 10,  # number of folds for cross-validation
       
        'collectStart': 5 * 60,  # seconds: when to start collecting the data
        'collectFor': 45 * 60,  # seconds: how long to collect data
       
        'nTrialBalance': 200,  # Balance the data (take nTrial of each behavior)
        'minActTime': 0.11,  # Only count (use) a behavior if it's at least this long (in seconds)
        'minNoRepeatTime': 0.59999,  # can't have occurred this much time in the past (in seconds)
      
        'minBhvNum': 20,  # must have occurred at least this many times to analyze
        'minOneBackNum': 40,  # For one-back (previous behavior) analyses
        'nOneBackKeep': 6,  # For one-back analyses
      
        'removeSome': True,  # Neural activity options
        'minFiringRate': 0.5,  # Minimum acceptable firing rate to include a neuron
        'maxFiringRate': 40,  # Maximum acceptable firing rate to include a neuron
    }
    return opts


"""Multi-object tracking -- ByteTrack integration. PLACEHOLDER, build step 4.

Gives detections persistent IDs across frames, which two downstream capabilities
require and cannot be built without:

  * **time-to-collision** needs the *same* object's box across consecutive frames
    to measure scale growth; per-frame detections alone cannot give a rate.
  * **lane-change detection** needs the ego vehicle's lateral offset tracked over
    time to tell a crossing from a one-frame lane-fit wobble.

ByteTrack over DeepSORT because it associates using detection confidence alone --
no appearance-embedding network, so no second model on the critical path, which
matters on an 8 GB laptop GPU already hosting the backbone and MiDaS.
"""

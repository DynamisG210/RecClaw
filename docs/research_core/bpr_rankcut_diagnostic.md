# BPR Rank-Cut Diagnostic Extension

BPR rank-cut is retained only as a diagnostic negative example.

Evidence summary:

- active rank-cut metric smoke regressed versus the BPRMargin parent
- disabled `rankcut_weight=0.0` path matched the parent metrics
- the active regression is therefore adjudicated as a mechanism-negative
  activation-metric mismatch, not a wrapper/integration blocker

Mainline policy:

- keep local diagnostic code and tests if useful
- do not promote to active candidate registry
- do not add to default action space
- do not claim metric or search-quality improvement

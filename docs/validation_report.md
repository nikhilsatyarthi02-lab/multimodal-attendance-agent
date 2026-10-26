# Validation report (Activity 6.3)

| # | Case | Result | Detail |
|---|------|--------|--------|
| 1 | Missing video file | PASS | raised FileNotFoundError as intended |
| 2 | Corrupted / non-video file | PASS | raised FileNotFoundError as intended |
| 3 | No faces in video (empty room) | PASS | completed; participants=0, present=0, warnings=['No audio track found; using vision signals only.', 'No faces detected; check camera angle, lighting and face size.'] |
| 4 | Video without audio track | PASS | completed; participants=0, present=0, warnings=['No audio track found; using vision signals only.', 'No faces detected; check camera angle, lighting and face size.'] |
| 5 | Very short video (0.5 s) | PASS | completed; participants=0, present=0, warnings=['No audio track found; using vision signals only.', 'No faces detected; check camera angle, lighting and face size.'] |
| 6 | Completely silent audio | PASS | completed; participants=2, present=2, warnings=[] |
| 7 | Constant loud noise | PASS | completed; participants=2, present=2, warnings=[] |
| 8 | Low lighting (35% brightness) | PASS | completed; participants=2, present=2, warnings=['No audio track found; using vision signals only.'] |
| 9 | Normal demo session (reference) | PASS | completed; participants=2, present=2, warnings=[] |

**9/9 cases passed.**

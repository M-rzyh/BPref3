# PEBBLE Framework Parameters Summary

## Parameter Definitions

- **`num_unsup_steps=9000`** - Number of unsupervised training steps before human feedback begins. The agent trains using only environment rewards, no human preferences.

- **`num_train_steps=500000`** - Total number of training steps for the entire run. Hard limit on when training terminates.

- **`num_interact=20000`** - Number of environment interactions between preference queries. Agent must interact with environment 20k times before requesting new human feedback.

- **`max_feedback=500`** - Hard limit on total number of preference comparisons/signals the algorithm can collect during entire training.

- **`reward_batch=50`** - Batch size for each preference query. Each time the algorithm queries for human feedback, it receives exactly 50 preference signals.

- **`reward_update=50`** - Number of gradient update steps when training the reward model per batch.

---

## Key Calculation: Feedback Query Schedule

**Formula:**
```
num_train_steps = num_unsup_steps + (number_of_feedbacks × num_interact)
```

**With current parameters:**
```
500,000 = 9,000 + (feedbacks × 20,000)
(500,000 - 9,000) / 20,000 = 24.55 ≈ 25 feedbacks
```

**Feedback query timeline:**
- Occurs at steps: ~29k, ~49k, ~69k, ~89k, ~109k, ~129k, ~149k, ~169k, ~189k, ~209k, ... (every 20k interactions)
- Approximately **25 queries** would happen before 500k steps

---

## How max_feedback Acts as a Hard Limit

`max_feedback` determines when preference collection **stops**, not which preferences get selected:

1. Queries happen at intervals (`num_interact=20k`)
2. Each query collects `reward_batch=50` preferences
3. Once total collected reaches `max_feedback=500`, **no more queries are requested**
4. Training continues to 500k steps with only 500 preferences collected

**With max_feedback=500:**
- Only 10 queries occur (500 / 50 = 10)
- These happen at steps: ~29k, ~49k, ~69k, ~89k, ~109k, ~129k, ~149k, ~169k, ~189k, ~209k
- After ~209k steps, no more preference queries despite training continuing

---

## Total Preference Signals Collected

- **Per query:** 50 preference signals
- **Number of queries (with current settings):** 10 queries (limited by `max_feedback=500`)
- **Total preferences collected:** 10 × 50 = **500 preference signals**

(If `max_feedback` weren't limiting, ~25 queries would happen, resulting in ~1,250 total preferences, but 500 is the hard stop)

---

## Summary of Training Flow

1. **Steps 0-9k:** Unsupervised learning (no human feedback)
2. **Steps 9k-29k:** Policy learning without feedback (after 20k interactions)
3. **Step ~29k:** First preference query (50 signals collected, total=50)
4. **Every 20k interactions thereafter:** New preference query until max_feedback limit
5. **After ~209k steps:** No more queries (500 max preferences reached)
6. **Steps 209k-500k:** Policy learning continues with fixed 500 collected preferences
7. **Step 500k:** Training terminates

---

## Key Insights

- `num_train_steps` and `num_interact` determine the *potential* number of queries
- `max_feedback` acts as a hard ceiling on preference collection
- Whichever constraint is hit first determines actual feedback count
- In this setup: **max_feedback (500) is the limiting factor**, not the step count
- The reward model trains on accumulated preferences using `reward_batch` and `reward_update` settings

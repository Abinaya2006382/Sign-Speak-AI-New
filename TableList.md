# Database Tables

## Table: gestures

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| gesture_name | TEXT | Gesture name |
| created_at | TIMESTAMP | Creation time |

---

## Table: history

| Column | Type | Description |
|---|---|---|
| id | INTEGER | Primary key |
| gesture | TEXT | Detected gesture |
| confidence | REAL | Prediction confidence |
| created_at | TIMESTAMP | Detection time |

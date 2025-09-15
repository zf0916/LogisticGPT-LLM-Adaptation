Streamlit fixes
1. missing label warning
-   File "C:\Users\Admin\Documents\text2sql\app\pages\sql_instructions.py", line 32, in display_instruction_tab
-- label="sql_instructions"

SQL Instruction
Minimal
Mandatory — use these in your chain-of-thought and final SQL.
1. If public.pickup or public.delivery are used, alias them as p and d.
2. Read-only queries (SELECT/CTE only).
3. Prefer the smallest number of joins needed.
4. Compute durations as “hours” (end − start).
5. Return the keys the question implies (e.g., city/courier/date) with the metric.
6. If selecting a single best/worst, sort deterministically and LIMIT 1.
7. Exclude clearly invalid/missing timestamps before aggregation.

Max
Mandatory — use these in your chain-of-thought and final SQL.
1. Tables & Aliases: Only use `public.pickup p` and `public.delivery d`. Always qualify with `public.` and use aliases `p` and `d` consistently.
2. Casing & Identifiers: SQL keywords UPPERCASE; identifiers `lower_snake_case`.
3. SELECT-only: Produce safe, read-only queries (no DDL/DML).
4. Minimal Joins: Minimize tables. Avoid joins unless required by the question.
5. Counting & Aggregates: Use `COUNT(*)` unless counting a specific subset (or `COUNT(DISTINCT ...)` when asked).
6. Averages & Floats: Cast averages to float: `AVG(x)::float`.
7. Durations (hours): `EXTRACT(EPOCH FROM (end_ts - start_ts)) / 3600.0`.
8. Day/Hour Buckets:
    - Day: `date_trunc('day', ts) AS day` (and if selecting the date value, use `day::date AS date`).
    - Hour: `EXTRACT(HOUR FROM ts) AS hour_of_day`.
9. Robustness Thresholds: When ranking groups (city/courier/region) and the question implies “reliable”/“typical”/“top/worst” cohorts, add `HAVING COUNT(*) >= 30`.
10. Tie-breakers: Whenever sorting by an aggregate, add a secondary deterministic sort on the entity/time key (e.g., `ORDER BY cnt DESC, city ASC` or `ORDER BY avg_hours ASC, courier_id ASC`).
11. LIMIT discipline: If the question asks for a single best/worst/earliest/latest, add `LIMIT 1` (after tie-breakers).
12. Anti-joins: Prefer `LEFT JOIN ... ON ... WHERE right.key IS NULL` over `NOT IN (SELECT ...)`.
13. CTEs & Scope: When you use CTEs/subqueries, refer to columns via the derived table alias in outer queries (never reuse inner table aliases outside their scope).
14. Invalid/Empty Values: Exclude invalid or empty values before aggregations.
15. Delivery time vs Cycle time (very important):
    - _Delivery time_ is from accept → completion: `(d.timestamp_utc - d.accept_ts_utc)` and enforce `d.timestamp_utc >= d.accept_ts_utc`.
    - _Cycle time_ is from pickup → delivery: `(d.timestamp_utc - p.timestamp_utc)` and enforce `d.timestamp_utc >= p.timestamp_utc` (and keep `d.timestamp_utc >= d.accept_ts_utc` when both matter).
16. Late pickups: If “late after promised window” is implied, require `p.time_window_end_utc IS NOT NULL AND p.timestamp_utc > p.time_window_end_utc`.
17. Percentages: For share of total across groups, use a window: `100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS percentage`.
18. Answer shape: Return the keys the question implies (e.g., for “max deliveries in a day per courier”, select `(courier_id, date, cnt)`, not just the count).
19. Averaging across days: When asked for “average X per day”, first build per-day rows, then average the per-day counts (don’t average totals over day-of-week groups directly).
05 Annotate Query-SQL Dataset

Query Crafting Methods:
1. expert/operational practice review
2. KPI dashboards/reports analysis
3. paraphrase expansion
4. schema-guided mapping for comprehensive coverage
5. ambiguity/unanswerable detection for robustness

Query Categories:
A) Volume & Count
- mirrors ops dashboards for daily/weekly volumes, peaks, and totals using simple aggregates and UTC calendar buckets on `timestamp_utc

B) Time & Delay
- Delivery timeliness is measured as **accept → completion** (`accept_ts_utc` → `timestamp_utc`)
- Pickup timeliness is **window adherence** (compare `pickup.timestamp_utc` to `pickup.time_window_end_utc`)
- Thresholded delivery-delay questions (e.g., “> 2 hours”) **must specify the threshold explicitly** (no default SLA)

C) Courier Performance
- Per-courier metrics cover volume and speed
- Cycle time** (`delivery.timestamp_utc − pickup.timestamp_utc`) is computed only for orders with both events and **attributed to the delivery courier** unless the question specifies otherwise

D) Failure / Exceptions
- With no NULL columns, exceptions are defined as **cross-table gaps** (e.g., pickup-only).
- Data-quality checks focus on **chronology violations**: delivery completion before acceptance, and (for orders with both) delivery completion before pickup

E) Spatial Distribution
- Spatial cuts by `city`, `region_id`, and `aoi_type` quantify demand distribution and service performance across locations, aligning with last-mile analysis practices.

F) Grounded Values
- Stress-test model handling of specific IDs and values.
- Queries explicitly reference real order_id, courier_id, region_id, aoi_id, or timestamps.
- Ensures SQL annotation pipeline correctly resolves literals.

Crafted Query Examples:
A1. On average, how many packages are picked up per day?  
A2. On which date was the highest number of deliveries completed?  
A3. During which hour of day do pickups peak?  
A4. Which day of week has the fewest deliveries on average?  
A5. Over the dataset period, how many total deliveries were completed?

B1. What is the average delivery time in each city?  
B2. How many pickups occurred after their window time?  
B3. Which city has the longest average pickup delay beyond the promised time window?  
B4. What is the longest observed delivery time for any package?  
B5. How many deliveries took more than 2 hours from acceptance to completion?

C1. Which courier delivered the most packages overall?  
C2. Which courier has the fastest average delivery time?  
C3. On average, how many deliveries per day does each courier complete?  
C4. What is the maximum number of deliveries in one day by a single courier?  
C5. Which delivery courier has the shortest average cycle time ?

D1. How many orders have a pickup event but no delivery event?  
D2. Which region has the highest share of pickup-only orders?  
D3. Which pickup courier has the most pickup-only orders?  
D4. How many events violate chronology rules?  
D5. How many packages were reported as damaged?

E1. Which city handled the highest volume of deliveries?  
E2. What percentage of deliveries were made in each AOI type?  
E3. In which city do deliveries take the longest on average?  
E4. Which `region_id` recorded the highest number of deliveries?  
E5. Which city has the highest pickup volume?

F1. How many deliveries were completed by courier 2382?
F2. For order 2115566, what was the time between pickup and delivery?
F3. How many pickups occurred in AOI ID 50?
F4. How many deliveries occurred in region 10?
F5. At what time was the delivery for order 2031782 completed?

Dataset value for further/better question craft
1. order_id:
- 2115566, 3143800, 1934416, 4194897, 724444 (both pickup and delivery)
- 5589648, 5995079, 5811253, 4927053, 5498519 (only pickup)
- 2031782, 4285071, 4056800, 3589481, 275232 (only delivery)
2. courier_id:
- 2382, 427, 4173, 2127, 3578
3. aoi_id
- 0 to 200
4. region_id
- 3 to 200
5. aoi_type
- 0 to 15
6. timestamp_ac_uts
- 2025-06-11 08:04:00, 2025-10-15 01:26:00, 2025-10-03 02:06:00, 2025-09-09 11:08:00, 2025-05-14 11:23:00

Query Dataset Amount:
1. Test set (Gold): 30 queries (5 per category).
- A–E: schema-based general queries.
- F: grounded value queries.
2. Example set: 12 queries (2 per category).
- Used as knowledge base training examples in Datrics pipeline.
- Keep simple/diverse, but avoid overlap with gold queries.
3. Train set (optional fine-tuning): 60 queries (10 per category).
- Generated expansions via paraphrase, structural variation, and value injection.
- Includes negative/unanswerable cases for robustness.

SQL Annotation Canonicalization Rules
- Schema & Aliases
    - Always qualify: `public.pickup p`, `public.delivery d`.
    - Use only `p` for pickup, `d` for delivery.
- Style & Casing
    - SQL keywords in **UPPERCASE**.
    - Identifiers in **lower_snake_case**.
- COUNT / Aggregates
    - Always `COUNT(*)` unless intentional subset (`COUNT(DISTINCT …)`).
    - `AVG(...)::float` for averages.
    - Use `/ 3600.0` (not `/ 3600`).
    - Add `HAVING COUNT(*) >= 30` for robustness filters when needed.
- Ordering & Limits
    - Add a **secondary tie-breaker** in `ORDER BY` (e.g., `city ASC, day ASC`).
    - For “top/worst single”, enforce `LIMIT 1`.
- Time & Date Handling
    - Day bucket: `date_trunc('day', ts) AS day`.
    - Hour-of-day: `EXTRACT(HOUR FROM ts) AS hour_of_day`.
    - Durations (hours): `EXTRACT(EPOCH FROM (end_ts - start_ts)) / 3600.0`.
    - Delivery validity: `WHERE d.timestamp_utc >= d.accept_ts_utc`.
- Joins & Structure
    - Always explicit `JOIN … ON …`.
    - CTEs with **short, semantic names**.
- Null & Unanswerable
-   Use `gold_sql: null` for truly unanswerable questions (don’t fabricate).

3. SQL Instruction (Datrics Pipeline)
Allows the pipeline to generate SQL adhering the following instruction:
- Mandatory use these INSTRUCTIONS in Chain-of-Thoughts:
    1. Minimize the number of tables used in each query. Avoid unnecessary joins or operations.
    2. Before applying aggregations (e.g., SUM, AVG), ensure empty or invalid values are excluded.
    3. Qualify tables as public.pickup p and public.delivery d, use aliases p and d
    4. Use COUNT(*), explicit JOIN ... ON ..., UPPERCASE keywords with lower_snake_case identifiers
    5. Express durations as EXTRACT(EPOCH FROM (end_ts - start_ts)) / 3600.0.
    6. Prefer LEFT JOIN ... IS NULL for anti-joins instead of NOT IN (SELECT ...).
    7. When using subqueries/CTEs, reference columns via the derived table alias in the outer query (e.g., use daily.courier_id), never inner aliases like d.courier_id outside their scope.
* Note the SQL instruction must be a single line string in the descriptor

---

Annotation Pipeline
1. Query-SQL Dataset format
- JSONL fields required for evaluation: query_id, db_id, nl_query, gold_sql.
2. Query-SQL annotating
- Query–SQL pairs (test = 30, example = 12, train = 60) were generated with ChatGPT.
- Generation was guided by:
    - Logistic domain reference (business rules & KPIs)
    - Database schema (pickup & delivery tables)
    - Canonicalization rules (see SQL Crafting Requirement)
3. Validation
- All generated SQL were executed against the Postgres database (t2sql_db).
- Each query was checked for correctness and adjusted if needed (e.g., alias consistency in C3).
- Unanswerable cases were explicitly marked with gold_sql: null.
Data Preparation
1. Download Datasets from LaDe `data\raw`
2. Preprocess with `scripts\data_preprocess.ipynb` to `data\processed`

Docker DB Setup
1. Install Docker Desktop
2. Build from dockerfile `docker-compose up -d`
- compost_stack: text2sql
- container: postgres_t2sql_db
- username/password: postgres (default)
3. Copy data files into container
```
docker cp data\processed\pickup_combined.csv postgres_t2sql_db:/tmp/pickup_combined.csv

docker cp data\processed\delivery_combined.csv postgres_t2sql_db:/tmp/delivery_combined.csv
```
4. Log into container with PostGres
- `docker exec -it postgres_t2sql_db psql -U postgres`
5. Create Database
- `CREATE DATABASE t2sql_db;`
6. Load into DB
- `\c t2sql_db`
7. Create Table
```
CREATE TABLE pickup (
  order_id             BIGINT,
  region_id            INT,
  city                 TEXT,
  courier_id           INT,
  lng                  DOUBLE PRECISION,
  lat                  DOUBLE PRECISION,
  aoi_id               INT,
  aoi_type             INT,
  timestamp_utc        TIMESTAMPTZ,
  accept_ts_utc        TIMESTAMPTZ,
  time_window_start_utc TIMESTAMPTZ,
  time_window_end_utc   TIMESTAMPTZ
);

CREATE TABLE delivery (
  order_id             BIGINT,
  region_id            INT,
  city                 TEXT,
  courier_id           INT,
  lng                  DOUBLE PRECISION,
  lat                  DOUBLE PRECISION,
  aoi_id               INT,
  aoi_type             INT,
  timestamp_utc        TIMESTAMPTZ,
  accept_ts_utc        TIMESTAMPTZ,
);
```
8. Copy files to tables
```
COPY pickup (
  order_id, region_id, city, courier_id, lng, lat, aoi_id, aoi_type,
  timestamp_utc, accept_ts_utc, time_window_start_utc, time_window_end_utc
)
FROM '/tmp/pickup_combined.csv'
WITH (FORMAT csv, HEADER true, NULL '');

COPY delivery (
  order_id, region_id, city, courier_id, lng, lat, aoi_id, aoi_type,
  timestamp_utc, accept_ts_utc
)
FROM '/tmp/delivery_combined.csv'
WITH (FORMAT csv, HEADER true, NULL '');
```

9. Quick Check
```
SELECT COUNT(*) AS pickup_rows FROM pickup;
SELECT COUNT(*) AS delivery_rows FROM delivery;
```

Webapp Schema Indexing
1. run `streamlit run main.py`
2. Go to Documentation tab

* Must upload one file at a time
3. Upload table schema (.md or .txt)

* If database index corrupted (failed to retrieve correct tables)
-> delete the files in the folder vector_db_storage and reupload schema and index
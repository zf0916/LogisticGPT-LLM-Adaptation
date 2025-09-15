This document is to record the steps to carry out a text2sql solution by Datrics Text2SQL on the dataset by CaiNiao/LaDe

Datrics Text2SQL: An open source text-to-SQL solution that utilize these functions:
1. Knowledge Base Training - Build database metadata based on the following:
- Database Schema
- Query-SQL Example
2. Content Retrieval - based on the knowledge base to retrieve relevant document and examples with reference to business rules incorporated
3. SQL Generation - use the retrieved document and examples to create the correct SQL query

LaDe: A dataset with 2 subset depicting courier pickup and delivery events across 5 cities in China over a period of 6 months

The main procedures are:

A. Setting up the base settings to run Datrics Text2SQL pipeline
1. Env Setup
- Environment
- Docker
- Streamlit

2. Database setup
- Data Preprocess
- Docker DB setup
- Webapp Schema Index

3. Descriptor
- Model Routing
- SQL Instruction
- Business Rules

B. Pipeline Setup
4. Evaluation Requirements
- Hardware metrics - latency + tokens + VRAM usage
- Result metrics - EM + ESM + ETM
- Predictions format
- Preds and Gold Format
- Run format

5. Annotate Query-SQL Dataset
- Query Crafting Methods
- Query Categories
- Crafted Query Examples
- Dataset value for further/better question craft
- Query Dataset Amount
- SQL Crafting Requirment
- Annotation Pipeline

6. Generation Pipeline
- Loading query set
- Adapt/Calling Datrics generation functions
- Output Predictions

7. Evaluation Pipeline
- Load predictions
- Evaluate predictions
- Display Results

C. Evaluation across different settings/model
7. Baseline model + default settings
- sql: gpt-4o, table gpt-4o-mini
- no examples
- default sql instructions
- no business rules

8. Baseline model + best settings
- sql: gpt-4o, table gpt-4o-mini
- Updated examples
- Updated sql instructions
- Updated business rules

9. Open Sourced model + best settings
- sql: sqlcoder, table sqlcoder

10. Fine tune model + best settings
- QLoRA, GPTQ
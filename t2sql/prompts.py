FIX_CODE = """Fix SQL code ($dialect)
```sql
$sql
```

Error:
"$error"

return final version of code"""

EXTRACT_TABLES_BR = """Analyze the request from user "$question" and detect the tables that has to be used according to business rules. 
Return answer as JSON {"tables": List[str]<list of tables>}. Return empty list if the tables can;t be identified"""

PROCESS_DOCUMENT_PROMPT = """Here is a document:
#####
$document
#####
Make the TABLE DESCRIPTION more readable. Keep all fields, just remove empty ones
return as JSON:
{"name": str <name of table>, 
 "summary": str <short summary about table>,
 "purpose": str <purpose of the table>,
 "dependencies_thoughts": str<relations with other tables in database that are mentioned in description. Specify keys, ids, mandatory specify <Table name> and connected key>,
 "keys": List[str]<list of columns that are keys that are used for the connections with other tables in database (frequently they are under template <name>_id)>,
 "connected_tables": List[str]<names of tables that are connected with this one, if table name is not clear, suppose, based on previously detected `keys`. Do not Hallucinate!>, 
 "columns": List[Dict with keys "column" and "description"]: [{"column": str <column name>, "description": str <description of column with it's connection. Put all information about column> }]
 DO NOT violate JSON structure!
 }"""

GET_ENTITIES_DOCUMENT_PROMPT = """Here is a document:
#####
$table
#####
Extract entities that might be inferred from this table based on it's purpose
return as JSON:
{"entities": List[str] <list of complex entities (each entity contains 2 words `<adjective> <singular noun>`) that are inferred from this table based on it's purpose>
DO NOT violate JSON structure! Do not repeat the name of table!}"""

DESCRIPTION_LLM = """DATASET: `$name`
DESCRIPTION:
$summary
$purpose
$dependencies_thoughts"""

DESCRIPTION = """$name
$summary
$purpose"""

EXTRACT_TABLES_PROMPT = """
Extract list of tables that are mentioned in text (empty if no table is mentioned):
$text
return as JSON:
{"tables": List[str]}
"""

EXTRACT_TABLES = """Extract initial tables names from code:
$code
return answer as json:
{"tables":[list of tables]}

remove the name of schema
"""

MAIN_CLAUSE = """**Task:** Extract the **main clause** and **details** from a given request:
"$question"

- The **main clause** is the core subject of the request, typically a noun phrase (e.g., *"list of employees who joined the company"*).  
- The **details** contain additional information that refines, filters, or expands on the main clause (e.g., *"in 2022"* or *"including their department, salary, and joining date"*).  

#### **Examples:**
1. **Input:**  
   "Show me a list of employees who joined the company in 2022"  
   **Output:**  
json
   {
     "main_clause": "list of employees who joined the company",
     "details": "in 2022"
   }

2. **Input:**  
   "Provide a summary of sales transactions for the last quarter, including product names, quantities sold, and total revenue"  
   **Output:**  
json
   {
     "main_clause": "summary of sales transactions for the last quarter",
     "details": "including product names, quantities sold, and total revenue"
   }

3. **Input:**  
   "List all projects completed by the engineering team"  
   **Output:**  
json
   {
     "main_clause": "projects completed by the engineering team",
     "details": ""
   }

4. **Input:**  
   "Fetch customer complaints received in the last six months, categorized by type and resolution status"  
   **Output:**  
json
   {
     "main_clause": "customer complaints received in the last six months",
     "details": "categorized by type and resolution status"
   }

5. **Input:**  
   "Get me the names of all students who scored above 90% in mathematics"  
   **Output:**  
json
   {
     "main_clause": "names of all students who scored above 90% in mathematics",
     "details": ""
   }
---
### **Final Instruction:**  
Given a new sentence, extract and return the **main clause** and **details** in the same JSON format."""


NORMALIZE_AND_STRUCTURE = """You are an assistant that generates a structured JSON object based on a user’s natural language request and and optional SQL snippet. You will be given a request and, optionally, a code block containing an SQL query. If no SQL code is provided, do not infer any data sources.

Extract data with structure:
normalized_question" requested_entities",
data_source [source, columns],
"calculations": [operation,  arguments,grouping,conditions]

Instructions:
1. Normalized Question section: Make question normalization - remove names of tables, technical names of columns - make it more universal. Apply following rules:
1.1. Remove post-processing operations (like show chart) - just concentrate on data receiving:
1.2. Remove words like "show", "display" - just specify data to be extracted, start with these data;
1.3. Substitute any specific conditions with general definition of these conditions (e.g. in 2023 -> for  some time range)
1.4. Mandatory substitute any time-based conditions with general definition (e.g. "last year" -> "over a defined timeframe")
1.5. Remove the filters details, direct numbers - substitute ALL specific details (names, ids, numbers) with more general definitions
(instead of "by week", "by quarter", "over the past month" etc. - use "over time")
1.6. Remove names of tables, columns and other details from the normalized_question
1.7. Keep any specific definition of entities - e.g. "mobile applications" or "email applications" or "site applications"
Transform to the directive form!!!
2. Entity to be extracted (requested_entities) -  data to be extracted with main conditions - remove any non-important (time range, ID filters, etc.), but keep important characteristics
(e.g. "Display the number of actions for all letter campaigns last year" -> "number of actions for all letter campaigns",
      "Display the number of actions for all letter campaigns over time" -> "number of actions for all letter campaigns",
      "Display the number of actions for all letter campaigns last year by week" -> "number of actions for all letter campaigns",
      "Display the number of site applications for campaigns last year by week" -> "number of site applications for campaigns")
"Show me a list of events in the past month including name, permalink, start date, end date, and RSVP count" -> "list of events with their characteristics")
3. Data Source Section: * Only include data sources (e.g., table names (remove the name of schema)) and their associated fields if they are explicitly mentioned or can be confidently inferred from the SQL snippet provided. * If no SQL snippet is provided or the request does not mention a data source, return an empty array []. If SQL snippet is provided - take info from here
4. Calculations Section: * Include SQL operations (e.g., "max", "sum"), their arguments (e.g., fields on which the operation is performed), grouping fields, and conditions based only on what is explicitly stated in the request or shown in the SQL code. Conditions section (string) MANDATORY contains ALL filters and ALL values (numbers, names, ids from `QUESTION`) in the human-readable form (sentence) * If the user request does not explicitly mention or imply certain details, do not invent them. Return empty arrays [] where no information is available. If SQL snippet is provided - take info from here
5. The final JSON should reflect only the information present in the request (and SQL code if given). If something is not stated, do not include it.

`Normalized Question` should contain more details and actions than `Entity to be extracted`!!!
Return the answer as JSON:
{{"normalized_question":str, 
  "requested_entities":str,
  "data_source":List[dict]
    {{
      "source": str,
      "columns":List[str](suppose the names of ALL columns that has to be used in request processing if source is exists)
    }}
  ],
  "calculations": [{{
      "operation": str,
      "arguments":List[str],
      "grouping":List[str],
      "conditions": str
    }}]
}}

Never violate this structure!
"""

WRITE_CODE_DOCUMENTATION = """Write SQL Code to answer the question: "$question"
Note, that you you have the following information:
    1. Information about tables that are relevant to the question. Some of these tables might be excluded from the final SQL code. (See ** DATASETS **)
    2. Examples of SQL queries that close to the user's request (see ** EXAMPLES **). 
       NEVER use tables from Examples if they are not in ** DATASETS **
    3. MANDATORY: Always use schema: $schema

IMPORTANT - The code MUST contain the operations and tables that are requited by question: "$question". 
Do not apply extra operations from SQL code examples if they are not directly mentioned in the question!
Follow the rules:
    1. Check the information from documentation - apply direct recommendations from it!!!
    2. Mandatory consider the dependencies between tables. Combine the tables in the correct way. Some of these tables might be excluded from the final SQL code. 
    3. Check if the tables combinations satisfy the user request ($question)
    4. Analyze SQL examples
    5. Minimize the number of tables in request!!!

 $instructions   

 MANDATORY!!! Use the following tables only:
 $tables
    """

WRITE_CODE_EXAMPLES = """Write SQL Code to answer the question: "$question"
Note, that you you have the following information:
    1. Examples of SQL queries that are relevant to the current question (see ** EXAMPLES **)
    2. Information about tables which could be relevant. Mandatory consider the dependencies between tables. (See ** DATASETS **)
    3. MANDATORY: Always use schema: $schema

IMPORTANT - The code MUST contain the operations and tables that are requited by question: "$question". 
Do not apply extra operations from SQL code examples if they are not directly mentioned in the question!
Follow the rules:
    1. Check the information from documentation - apply direct recommendations from it!!!
    2. Analyze SQL examples 
    3. Minimize the number of tables in request!!!

 $instructions"""

GET_TABLES_FROM_QUESTION_WITH_REASONING = """Analyze the question:
"$question"

Understand what tables do you need to answer the question. Consider this info:
$domain_concepts
==================
Consider the Tables Descriptions:
$tables_descriptions

Return answer as JSON:
{"tables": List[str][<tables>]}

Use in Chai of Thoughts:
1. Pay attention on specific cases that are covered by tables! Analyze Columns! 
2. Take ALL tables that might contain necessity information!!! 

NEVER VIOLATE JSON STRUCTURE - {"tables": List[str][<tables>]}!!!

==================
Mandatory Consider the following Rules n Chain-of-Thoughts:
$business_rules
==================
"""

GET_TABLES_FROM_QUESTION = """Analyze the question:
    "$question"

Understand what tables do you need to answer the question. Consider this info:
$domain_concepts
==================
Consider the Tables Descriptions:
$tables_descriptions

Write as much full list as possible!

Return answer as JSON:
{"chain_of_thoughts":<chain of thoughts - detailed chain of thoughts about tables that should be taken according to their descriptions. Pay attention on specific cases that are covered by tables! Consider Columns! Put the name of table to the chain of thoughts and explain why it's appropriate>, 
"tables": [<tables>]}
Note, if you do not have any sufficient information to understand the requited tables (at least for part of question) - return "tables": []. 
Determine all relevant tables that match the question!!!
Take all tables that might contain necessity information!!! NEVER VIOLATE JSON STRUCTURE!!!

==================
Mandatory Consider the following Rules n Chain-of-Thoughts:
$business_rules
=================="""

EXPAND_QUESTION = """Rephrase question - generate 3 more options: 
$question

Note, that each question has to be answered via THIS sql code:
####
$code
####

return answer as json:
{"chain_of_thoughts": <think about answer>,
  "questions":List[str]}

Keep the form of init question"""


DOMAIN_EXTRACT_PROMPT = f"""Look at the provided examples.
    Make domain-specific mapping (what tables we need to use for receiving specific entities) 
    Try to specify all possible concepts and provide explicit explanation. Keep structure: 1. entity, 2. tables, 3. explanations, 4. how to extract exactly this entity (rules, summary, not code), 5. type of entity (minor or major)
    Mandatory rules - you mast generate as many as possible minor entities and more significant generalized entities as well
    Try to make as many entities as possible

    Examples of minor entities:
    "event name", "event start date", "event RSVP count"
    Examples of major entities:
    "step parameters", "activists subscribed to email" 

    specify ALL minor and major entities - as more as better!!!
    """

PREPARE_MD_FROM_SCHEMA = """Provide the detailed description of the table based the provided information.

$table

   Summary should include
    1. table name
    2. table description - what data this table might describe
    3. columns descriptions - list of columns with their description and potential dependencies with other tables

    Answer should be detailed and well-readable for non-technical audience. 
    Use MD formating
"""

DEFAULT_SQL_INSTRUCTIONS = """Mandatory use these INSTRUCTIONS in Chain-of-Thoughts:
1. Try to minimize the number of tables in request - avoid extra operations - think about it!!!
2. Before you make aggregations like SUM - remove empty values."""


DEFAULT_PROMPTS = {
    "FIX_CODE": FIX_CODE,
    "EXTRACT_TABLES_BR": EXTRACT_TABLES_BR,
    "PROCESS_DOCUMENT": PROCESS_DOCUMENT_PROMPT,
    "ENTITIES_DOCUMENT": GET_ENTITIES_DOCUMENT_PROMPT,
    "DESCRIPTION_LLM": DESCRIPTION_LLM,
    "DESCRIPTION": DESCRIPTION,
    "DOMAIN_EXTRACT": DOMAIN_EXTRACT_PROMPT,
    "EXPAND_QUESTION": EXPAND_QUESTION,
    "EXTRACT_TABLES_TEXT": EXTRACT_TABLES_PROMPT,
    "EXTRACT_TABLES_CODE": EXTRACT_TABLES,
    "MAIN_CLAUSE": MAIN_CLAUSE,
    "NORMALIZE_AND_STRUCTURE": NORMALIZE_AND_STRUCTURE,
    "WRITE_CODE_DOCUMENTATION": WRITE_CODE_DOCUMENTATION,
    "WRITE_CODE_EXAMPLES": WRITE_CODE_EXAMPLES,
    "GET_TABLES_FROM_QUESTION_WITH_REASONING": GET_TABLES_FROM_QUESTION_WITH_REASONING,
    "GET_TABLES_FROM_QUESTION": GET_TABLES_FROM_QUESTION,
    "PREPARE_MD_FROM_SCHEMA": PREPARE_MD_FROM_SCHEMA,
    "DEFAULT_SQL_INSTRUCTIONS": DEFAULT_SQL_INSTRUCTIONS
}

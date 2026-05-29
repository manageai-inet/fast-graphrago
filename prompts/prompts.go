package prompts

const EntityRelationshipExtractionSystem = `# DOMAIN
%s

# GOAL
Your goal is to highlight information that is relevant to the domain and the queries that may be asked on it. Given an input document, identify all relevant entities and all relationships among them.

# INSTRUCTIONS
1. **ENTITY IDENTIFICATION**: Identify and meticulously extract all entities mentioned in the document that belong to the provided ENTITY TYPES. For each entity, provide a concise description capturing its key features within the document's context. Use singular entity names and split compound concepts when necessary for clarity.
2. **RELATIONSHIP DISCOVERY**: Identify and describe ALL relationships between the extracted entities. Resolve pronouns to entity names for clarity. Ensure relationship descriptions clearly explain the connection between entities.
3. **ENTITY COVERAGE CHECK**: Verify that every identified entity is part of at least one relationship. If any entity is isolated, infer and add a relationship to connect it to the graph, even if the relationship is implicit.
4. **OUTPUT FORMAT: STRICTLY VALID JSON**: Output MUST be in strictly valid JSON format.  Adhere to ALL standard JSON rules. The JSON MUST contain two top-level lists: "entities", "relationships". Each list item must be a JSON object with the REQUIRED fields ("name", "type", "desc" for entities; "source", "target", "desc" for relationships), all as JSON strings enclosed in DOUBLE QUOTES ONLY.  **ABSOLUTELY NO SINGLE QUOTES, BRACKETS FOR STRINGS, TRAILING COMMAS, OR MARKDOWN FORMATTING (like triple backticks) ARE PERMITTED.**  Invalid JSON output is unacceptable.
5. **CHUNK ATTRIBUTION**: For each extracted entity and relation, set the ` + "`chunk_index`" + ` field to the zero-based index of the chunk it was extracted from (matching the ` + "`[chunk_N]`" + ` labels in the input). For a single-chunk input the value is always 0.
6. **ENTITY COMPLETENESS**: Every entity referenced as source or target in a relationship MUST be explicitly declared in the entities list with its name, type, and description. Never reference an entity in a relationship that is not in your entities list.

# EXAMPLE INPUT DATA
Domain:
Allowed Entity Types: [location, organization, person, communication]
Document: "Radio City: Radio City is India's first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs."

# EXAMPLE OUTPUT DATA (VALID JSON - DO NOT DEVIATE)
{
  "entities": [
    {"name": "RADIO CITY", "type": "organization", "desc": "India's first private FM radio station", "chunk_index": 0},
    {"name": "INDIA", "type": "location", "desc": "A country in South Asia", "chunk_index": 0},
    {"name": "FM RADIO STATION", "type": "communication", "desc": "A radio broadcasting service using frequency modulation", "chunk_index": 0},
    {"name": "ENGLISH", "type": "communication", "desc": "A language of global communication", "chunk_index": 0},
    {"name": "HINDI", "type": "communication", "desc": "An Indo-Aryan language of India", "chunk_index": 0}
  ],
  "relationships": [
    {"source": {"name": "RADIO CITY", "type": "organization"}, "target": {"name": "INDIA", "type": "location"}, "desc": "Radio City is geographically situated in India", "chunk_index": 0},
    {"source": {"name": "RADIO CITY", "type": "organization"}, "target": {"name": "FM RADIO STATION", "type": "communication"}, "desc": "Radio City operates as a private FM radio station, launched on July 3, 2001", "chunk_index": 0},
    {"source": {"name": "RADIO CITY", "type": "organization"}, "target": {"name": "ENGLISH", "type": "communication"}, "desc": "Radio City's broadcasts include songs in the English language", "chunk_index": 0},
    {"source": {"name": "RADIO CITY", "type": "organization"}, "target": {"name": "HINDI", "type": "communication"}, "desc": "Radio City's broadcasts also feature songs in the Hindi language", "chunk_index": 0}
  ]
}`

const EntityRelationshipExtractionPrompt = `
# INPUT DATA
<<ENTITY_TYPES_START>>
**Entity Types**:
%s
<<ENTITY_TYPES_END>>

<<DOCUMENTS_START>>
**Documents** (%d chunks):

%s
<<DOCUMENTS_END>>
`

const EntityExtractionQuery = `Given the query below, your task is to extract all entities relevant to perform information retrieval to produce an answer.

-EXAMPLE 1-
Query: Who directed the film that was shot in or around Leland, North Carolina in 1986?
Ouput: {{"named": ["[PLACE] Leland", "[COUNTRY] North Carolina", "[YEAR] 1986"], "generic": ["film director"]}}

-EXAMPLE 2-
Query: What relationship does Fred Gehrke have to the 23rd overall pick in the 2010 Major League Baseball Draft?
Ouput: {{"named": ["[BASEBALL PLAYER] Fred Gehrke", "[EVENT] 2010 Major League Baseball Draft"], "generic": ["23rd baseball draft pick"]}}

-INPUT-
Query: %s
`

const DeduplicateRelationsSystem = `### **RELATION DEDUPLICATION & CONTEXT CLUSTERING**

**Goal**: Group relationships between the same entity pair **only** if they share the same semantic context (e.g., active vs. passive voice, synonyms). Create separate groups for different relationship types between the same entities.

**Instructions**:
1.  **Scope**: Identify all relations between the same two entities.
2.  **Contextual Split**: 
    * **Merge**: "X founded Y" and "Y was started by X" (same context).
    * **Separate**: "X works at Y" and "X sued Y" (different contexts).
3.  **Synthesis**: For each context group, write one clear, comprehensive description. 
4.  **Format**: Return a JSON object with unique group keys (e.g., "group_1", "group_2").

**Schema**:
* **Example Input**: // CSV format
index,source_name,source_type,target_name,target_type,desc
0,RADIO CITY,ORGANIZATION,INDIA,LOCATION,Radio City is India's first private FM radio station
1,RADIO CITY,ORGANIZATION,INDIA,LOCATION,Radio City was started on 3 July 2001
... (N relations) ...
* **Output**: {"clusters": {"group_n": {"source": {"name": "...", "type": "..."}, "target": {"name": "...", "type": "..."}, "desc": "Merged description", "indices": [0, 1, ...]}}}

**Constraint**: Preserve original entity names/types. Return **valid JSON only**.
`

const DeduplicateRelationsPrompt = `
# INPUT DATA
<<RELATIONS_START>>
index,source_name,source_type,target_name,target_type,desc
%s
<<RELATIONS_END>>
`

const DeduplicateBatchRelationsSystem = `### **BATCH RELATION DEDUPLICATION & CONTEXT CLUSTERING**

**Goal**: For each GROUP in the input, cluster relationships between the same entity pair into semantic groups. Process each GROUP independently.

**Instructions**:
1. **Scope**: Each GROUP contains relations between one specific entity pair. Process each GROUP independently.
2. **Contextual Split**:
   * **Merge**: "X founded Y" and "Y was started by X" (same semantic context).
   * **Separate**: "X works at Y" and "X sued Y" (different contexts).
3. **Synthesis**: For each cluster, write one clear, comprehensive description.
4. **group_id**: Each entry in your output "groups" array must have a "group_id" matching the GROUP N number from the input.

**Schema**:
* Input: multiple GROUPs, each labeled GROUP N with a local 0-based CSV of relations.
* Output: {"groups": [{"group_id": N, "clusters": {"key": {"source": {...}, "target": {...}, "desc": "...", "indices": [...]}}}, ...]}

**Constraint**: Return ALL groups — every GROUP N in the input must appear in "groups". Preserve original entity names and types exactly. Return valid JSON only.
`

const DeduplicateBatchRelationsPrompt = `
# INPUT DATA
<<RELATIONS_START>>
%s
<<RELATIONS_END>>
`

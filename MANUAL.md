# fast-graphrago — คู่มืออธิบายการทำงาน

> **fast-graphrago** คือ Go library สำหรับทำ **Graph RAG (Retrieval-Augmented Generation)** โดยใช้ Knowledge Graph ร่วมกับ Personalized PageRank ในการค้นหาข้อมูลที่เกี่ยวข้องก่อนส่งให้ LLM ตอบคำถาม

---

## สารบัญ

1. [Concept หลัก](#1-concept-หลัก)
2. [Architecture Overview](#2-architecture-overview)
3. [โครงสร้างไฟล์](#3-โครงสร้างไฟล์)
4. [Data Model](#4-data-model)
5. [Indexing Pipeline (การสร้าง Index)](#5-indexing-pipeline-การสร้าง-index)
6. [Retrieval Pipeline (การค้นหา)](#6-retrieval-pipeline-การค้นหา)
7. [Algorithm: Personalized PageRank (PPR)](#7-algorithm-personalized-pagerank-ppr)
8. [LLM Prompts](#8-llm-prompts)
9. [Configuration & Options](#9-configuration--options)
10. [การ Integrate เข้าโปรเจกต์](#10-การ-integrate-เข้าโปรเจกต์)

---

## 1. Concept หลัก

### RAG คืออะไร?

RAG (Retrieval-Augmented Generation) คือเทคนิคที่ให้ LLM ตอบคำถามโดยอ้างอิงจากข้อมูลที่ค้นหามาก่อน แทนที่จะพึ่งแค่ความรู้ที่ฝึกมา วิธีนี้ทำให้ LLM ตอบได้แม่นยำขึ้นและอ้างอิงข้อมูลจริงได้

### Graph RAG แตกต่างจาก RAG ทั่วไปอย่างไร?

| RAG ทั่วไป (Vector RAG) | Graph RAG |
|---|---|
| แปลงเอกสารเป็น vector แล้วค้นหาด้วย cosine similarity | สร้าง Knowledge Graph จากเอกสาร (entities + relations) |
| ได้ text chunk ที่ "ใกล้เคียง" query มากที่สุด | ได้ context ที่ "เชื่อมโยงกัน" ผ่าน graph |
| ดีสำหรับคำถามตรงๆ | ดีสำหรับคำถามที่ต้องการ reasoning ข้ามหัวข้อ |
| ไม่เข้าใจความสัมพันธ์ระหว่าง concept | เข้าใจว่า A → B → C เชื่อมกันอย่างไร |

### Knowledge Graph คืออะไร?

Knowledge Graph คือโครงสร้างข้อมูลแบบกราฟที่ประกอบด้วย:

- **Entity (Node)** — สิ่งที่มีอยู่ในเอกสาร เช่น บุคคล บริษัท สถานที่ แนวคิด
- **Relation (Edge)** — ความสัมพันธ์ระหว่าง entity เช่น "นาย ก. **ทำงานที่** บริษัท ข."

```
[นาย ก.]──ทำงานที่──>[บริษัท ข.]──ตั้งอยู่ที่──>[กรุงเทพ]
    │
    └──เป็นผู้จัดการของ──>[โปรเจกต์ ค.]──ใช้เทคโนโลยี──>[Go]
```

### Personalized PageRank (PPR) คืออะไร?

PPR คือ algorithm ที่ใช้หาว่า node ไหนใน graph "สำคัญ" ที่สุดสำหรับ query นั้นๆ โดยจำลอง random walker ที่เริ่มต้นจาก node ที่ตรงกับ query มากที่สุด แล้ว "เดิน" ไปตาม edges ของ graph

ผลลัพธ์คือ ranking ของ entities ทั้งหมด โดย entity ที่อยู่ใกล้กับ seed (query entities) มากที่สุดจะได้ score สูงสุด

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    ผู้ใช้ (Caller)                    │
└────────────────────┬────────────────┬────────────────┘
                     │                │
              Index(docs)      Retrieve(query)
                     │                │
┌────────────────────▼────────────────▼────────────────┐
│              FastGraphIndexer  (indexer.go)           │
│         Public API: Index() / Retrieve()              │
└────────────────────┬────────────────┬────────────────┘
                     │                │
┌────────────────────▼────────────────▼────────────────┐
│           GraphRAGServiceImpl  (rag/graph_rag.go)     │
│     แกนกลางของระบบ — ประสาน Extraction, Embedding,    │
│     Graph Construction, PageRank ทั้งหมด               │
└──┬──────────────┬──────────────┬───────────────┬─────┘
   │              │              │               │
   ▼              ▼              ▼               ▼
FastGraph    ChunkExtractor  VectorStore   AssetStore
Extractor    (chunking)      (vectors)    (graph data)
(graph/)
   │
   ▼
LLM (Tool Calling)
```

### Component หลัก

| Component | ไฟล์ | หน้าที่ |
|---|---|---|
| `FastGraphIndexer` | `indexer.go` | Public API wrapper |
| `GraphRAGServiceImpl` | `rag/graph_rag.go` | Logic หลักทั้งหมด |
| `FastGraphExtractor` | `graph/extractor.go` | ใช้ LLM ดึง entity/relation |
| `LLM` | `llm/interface.go` | Interface สำหรับ Chat Completion |
| `OpenAIEmbedder` | `llm/embedder.go` | Batch embedding ด้วย OpenAI |
| Models | `models/graph.go` | Data structures ทั้งหมด |
| Options | `rag/options.go` | Configuration ทั้งหมด |

---

## 3. โครงสร้างไฟล์

```
fast-graphrago/
├── indexer.go              ← Public API: New(), Index(), Retrieve()
├── models/
│   └── graph.go            ← Data structs: GraphEntity, GraphRelation, TransitionMatrix
├── graph/
│   ├── interface.go        ← GraphExtractor interface
│   └── extractor.go        ← FastGraphExtractor: LLM-based extraction
├── rag/
│   ├── graph_rag.go        ← GraphRAGServiceImpl: ตรรกะหลักทั้งหมด
│   ├── options.go          ← Options pattern: WithDomain(), WithTopK() ฯลฯ
│   ├── interface.go        ← GraphRAGService interface
│   └── graph_rag_test.go   ← Tests
├── llm/
│   ├── interface.go        ← LLM interface
│   ├── openai.go           ← OpenAI implementation
│   └── embedder.go         ← OpenAIEmbedder
├── prompts/
│   └── prompts.go          ← System/user prompts สำหรับ LLM
├── utils/
│   ├── const.go            ← Default values ทั้งหมด
│   └── tools.go            ← JSON schema helpers
└── example/                ← ตัวอย่าง HTTP server ด้วย Fiber
```

---

## 4. Data Model

### Asset Hierarchy

ในระบบนี้ทุกอย่างเก็บเป็น **Asset** ซึ่งมี hierarchy ดังนี้:

```
Document (1 ไฟล์)
├── Page (1 หน้า)
│   └── Chunk (ข้อความย่อย ~500 chars)
│       ├── Entity (node ใน graph)  ← มี vector embedding
│       └── Relation (edge ใน graph)
```

แต่ละ Asset มี `Refs` เชื่อมถึงกัน ด้วย ref types:
- `parent` — ชี้ขึ้น (chunk → page → document)
- `child` — ชี้ลง (document → page → chunk)
- `source` / `target` — สำหรับ relation ชี้ไปหา entity ต้นทาง/ปลายทาง

### โครงสร้างข้อมูลสำคัญ

```go
// Node ใน Knowledge Graph
type GraphEntity struct {
    Name       string  // ชื่อ entity (normalized to lowercase)
    Type       string  // ประเภท: person, organization, location, ...
    Desc       string  // คำอธิบาย
    ChunkIndex int     // อยู่ใน chunk ไหน
}

// Edge ใน Knowledge Graph
type GraphRelation struct {
    Source     RelativeEntity  // ต้นทาง
    Target     RelativeEntity  // ปลายทาง
    Desc       string          // อธิบายความสัมพันธ์
    ChunkIndex int
}

// Sparse matrix สำหรับ PageRank
type TransitionMatrix struct {
    CSR              *sparse.CSR        // Compressed Sparse Row matrix
    EntityIndiceToId []string           // index → assetId
    EntityIDToIndex  map[string]int     // assetId → index
    Dim              int                // จำนวน entity ทั้งหมด
}
```

### Entity Types ที่รองรับโดย default

```
person, organization, location, event, date, product,
technology, concept, document, project, role, process,
regulation, communication, goal, metric, challenge,
solution, resource, outcome, table, image, form
```

---

## 5. Indexing Pipeline (การสร้าง Index)

เมื่อเรียก `indexer.Index()` ระบบจะทำงานตามขั้นตอนนี้:

```
Input: []KnowledgeSource (page-level content)
         │
         ▼
[Step 1] preparePageAssets
         สร้าง Page Asset จากแต่ละหน้า
         assetId format: "kbId:filename:pageNumber"
         │
         ▼
[Step 2] ChunkingExtractor.Extract
         ตัดข้อความในแต่ละหน้าเป็น Chunk
         ขนาด default: 500 chars, overlap: 50 chars
         chunkId format: "kbId:filename:page[start:end]"
         │
         ▼
[Step 3] GraphExtractor.ExtractGraph  ← ใช้ LLM
         ┌──────────────────────────────────────────┐
         │ แบ่ง chunks เป็น batch (default: 5 chunks) │
         │ แต่ละ batch ส่งให้ LLM พร้อมกัน (concurrent)│
         │                                           │
         │ LLM → entity list + relation list         │
         │ normalize ชื่อ entity (lowercase)          │
         │ merge entity ที่ซ้ำกันข้าม batch           │
         │ deduplicate relation ที่ซ้ำกัน (LLM อีกรอบ)│
         └──────────────────────────────────────────┘
         │
         ▼
[Step 4] batchEmbed
         embed entity แต่ละตัวเป็น vector
         batch size default: 32 entities/call
         มี retry logic: 3 ครั้ง, delay 500ms
         fallback: embed ทีละตัวถ้า batch fail
         │
         ▼
[Step 5] VectorStore.InsertBatchVectors
         เก็บ vector embedding ของ entity
         │
         ▼
[Step 6] AssetStore.InsertBatchAssets
         เก็บ assets ทั้งหมด:
         [document, pages, chunks, entities, relations]
         │
         ▼
Output: IndexingResult (kbId, version, asset counts)
```

### Version Management

ทุกครั้งที่ Index ระบบจะสร้าง `version = time.Now().UnixMilli()` ใหม่เสมอ ทำให้สามารถ retrieve ข้อมูลจาก version ใดก็ได้ หรือ version ล่าสุดถ้าไม่ระบุ

### Relation Deduplication

เมื่อ LLM ดึง relation จากหลาย chunk อาจได้ relation เดียวกันซ้ำหลายครั้ง (เช่น "A ทำงานที่ B" ปรากฏใน 3 chunk) ระบบจะรวม relation กลุ่มเดียวกันเข้าหากันด้วยการส่งให้ LLM cluster แบบ batch (`DeduplicateRelationsBatch`) เพื่อลด LLM call

---

## 6. Retrieval Pipeline (การค้นหา)

เมื่อเรียก `indexer.Retrieve()` ระบบจะทำงานดังนี้:

```
Input: query string, kbIds []string, seedAssets
         │
         ▼
[Step 1] ExtractEntitiesFromQuery  ← ใช้ LLM
         แยก query ออกเป็น 3 ประเภท:
         • named entities  → "นาย ก.", "บริษัท ข."
         • generic entities → "ผู้จัดการ", "เทคโนโลยี AI"
         • query itself    → ทั้ง query string
         │
         ▼
[Step 2] EmbedContent (concurrent)
         embed query entities แต่ละตัวเป็น vector
         │
         ▼
[Step 3] getTransitionMatrixForKbId
         ดึง entity ทั้งหมดของ kbId จาก AssetStore
         สร้าง adjacency matrix (DOK → CSR):
           matrix[i][j] = จำนวน relation จาก entity_i → entity_j
         apply log1p weight clipping:
           w = log(1 + count)  ← ลด dominance ของ node ที่มี degree สูง
         normalize แต่ละแถวให้รวมเป็น 1.0 (stochastic matrix)
         │
         ▼
[Step 4] getSeedVector
         vector search หา entity ที่ใกล้กับ query entities
         คำนวณ seed vector s[] โดย:
           s[entity_idx] += score * weight

         weight ตาม entity type:
           named entity   → 0.35 / count_named
           generic entity → 0.15 / count_generic
           query itself   → 0.50

         normalize s[] ให้รวมเป็น 1.0
         │
         ▼
[Step 5] ppr()  ← Personalized PageRank
         power iteration 100 รอบ (default):
           v_next = v × P          (random walk)
           v = α×v_next + (1-α)×s  (teleportation)
         α (damping factor) = 0.5 default
         │
         ▼
[Step 6] rank entities ตาม PPR score
         เลือก Top-K entities (default: 50)
         filter threshold ≥ 0
         │
         ▼
[Step 7] propagate (concurrent per kbId)
         จาก top-K entities ดึงสิ่งที่เกี่ยวข้อง:
         • Relations ที่ชี้ไปหา entity เหล่านี้
         • Chunks ที่ entity และ relation อ้างถึง
         │
         ▼
Output: []RetrievedAsset (entities+score) + []ContextualAsset (relations+chunks)
```

### Query Entity Weights อธิบายเพิ่มเติม

```
QueryWeights = [0.35, 0.15, 0.50]
                  │      │      └── น้ำหนักของ query ทั้งประโยค
                  │      └───────── น้ำหนักรวมของ generic entities
                  └──────────────── น้ำหนักรวมของ named entities

ถ้า query มี named entities 2 ตัว → แต่ละตัวได้ 0.35/2 = 0.175
```

---

## 7. Algorithm: Personalized PageRank (PPR)

### ทำไมต้องใช้ PPR?

Vector search ธรรมดาหา entity ที่ "คล้าย" query มากที่สุด แต่ไม่รู้ว่า entity นั้นเชื่อมกับ entity อื่นๆ อย่างไร PPR แก้ปัญหานี้โดยเดินตาม graph โดยเริ่มจากจุดที่ใกล้ query มากที่สุด

### Transition Matrix

```
สมมติมี entity 3 ตัว:  A, B, C
relation: A→B (3 ครั้ง), A→C (1 ครั้ง), B→C (2 ครั้ง)

DOK matrix (raw count):
    A    B    C
A [ 0    3    1  ]
B [ 0    0    2  ]
C [ 0    0    0  ]

หลัง log1p weight clipping:
    A    B         C
A [ 0    log(4)   log(2)  ]  → normalize → [ 0    0.585   0.415 ]
B [ 0    0        log(3)  ]  → normalize → [ 0    0       1.0   ]
C [ 0    0        0       ]  → ไม่มี edge ออก (จะไม่ contribute)
```

### Power Iteration

```
กำหนด:
  s = seed vector (จาก vector search)
  P = transition matrix (stochastic)
  α = damping factor (0.5)
  v_0 = s

แต่ละ iteration:
  v_next[j] = Σ v[i] × P[i][j]   (random walk step)
  v[i]      = α × v_next[i] + (1-α) × s[i]  (teleportation)

หลัง 100 iteration → v คือ PPR score ของแต่ละ entity
```

### ความหมายของ Damping Factor (α)

- **α สูง (เช่น 0.85)** → random walker เดินตาม graph เยอะ → ผลลัพธ์ขึ้นกับ graph structure มาก
- **α ต่ำ (เช่น 0.3)** → random walker กลับมาที่ seed บ่อย → ผลลัพธ์ใกล้เคียง vector search
- **default = 0.5** → balance ระหว่างสองแบบ

### Log1p Weight Clipping

```
ทำไมต้องใช้ log(1 + count) แทน count ธรรมดา?

ถ้า entity A มี 100 relation ออก และ entity B มี 2 relation ออก
โดยไม่ clip: A dominate ทุก random walk
หลัง log1p: log(101) ≈ 4.6 vs log(3) ≈ 1.1  → ลด dominance ลงมาก
```

---

## 8. LLM Prompts

ระบบใช้ LLM ผ่าน **Tool Calling** (Structured Output) ใน 3 ขั้นตอน:

### Prompt 1: Graph Extraction (Indexing)

```
System: กำหนด domain + entity types
User:   ส่ง chunks (format: [chunk_0]\n...\n[chunk_1]\n...)

LLM return tool call:
{
  "entities": [
    {"name": "RADIO CITY", "type": "organization", "desc": "...", "chunk_index": 0}
  ],
  "relationships": [
    {"source": {...}, "target": {...}, "desc": "...", "chunk_index": 0}
  ]
}
```

**หมายเหตุ**: `chunk_index` ใช้บอกว่า entity/relation นั้นมาจาก chunk ไหน เพื่อ link กลับไปหา source

### Prompt 2: Relation Deduplication (Indexing)

```
System: อธิบาย goal: cluster relation ที่ semantic เหมือนกัน
User:   CSV ของ relations ทั้งหมดระหว่าง entity pair เดียวกัน

LLM return:
{
  "clusters": {
    "group_1": {
      "source": {...}, "target": {...},
      "desc": "merged description",
      "indices": [0, 2, 5]   ← relation index ที่ merge เข้ากัน
    }
  }
}
```

### Prompt 3: Query Entity Extraction (Retrieval)

```
User: "Who directed the film shot in Leland, North Carolina in 1986?"

LLM return:
{
  "named_entities": ["[PLACE] Leland", "[COUNTRY] North Carolina", "[YEAR] 1986"],
  "generic_entities": ["film director"]
}
```

---

## 9. Configuration & Options

### สร้าง Indexer

```go
indexer, err := fastgraphrag.New(
    llm,        // LLM implementation (ต้องรองรับ tool calling)
    vecStore,   // VectorStorage implementation
    assetStore, // AssetStorage implementation
    // Options เพิ่มเติม:
    rag.WithDomain("healthcare", []string{"person", "disease", "drug", "treatment"}),
    rag.WithBatchSize(10),          // chunks per LLM call
    rag.WithMaxConcurrent(10),      // concurrent LLM calls
    rag.WithEmbedder(embedder),     // batch embedder (OpenAIEmbedder)
    rag.WithEmbedBatchSize(64),     // entities per embed call
    rag.WithTopK(100),              // entity candidates จาก PPR
    rag.WithDampingFactor(0.7),     // α สำหรับ PPR
    rag.WithMaxWalks(200),          // iterations ของ PPR
)
```

### Runtime Options (ส่งตอน Index/Retrieve)

```go
config := map[string]any{
    // สำหรับ Index
    "domain":       "finance",
    "entity_types": []string{"person", "organization", "product"},

    // สำหรับ Retrieve
    "top_k":          50,
    "threshold":      0.3,
    "damping_factor": 0.5,
    "query_weights":  [3]float32{0.35, 0.15, 0.50},
    "version":        &version,   // ระบุ version ที่ต้องการ
    "label":          &label,     // filter ด้วย label
}
```

### Default Values

| Parameter | Default | ความหมาย |
|---|---|---|
| `Domain` | `"enterprise"` | Domain สำหรับ LLM extraction |
| `BatchSize` | `5` | chunks per LLM extraction call |
| `EmbedBatchSize` | `32` | entities per embedding call |
| `MaxConcurrent` | `5` | concurrent goroutines |
| `MaxWalks` | `100` | PPR iterations |
| `DampingFactor` | `0.5` | PPR alpha |
| `TopK` | `50` | entities ที่ return จาก retrieval |
| `Threshold` | `0.3` | minimum similarity score |
| `QueryWeights` | `[0.35, 0.15, 0.50]` | named / generic / query |
| `ChunkSize` | `500` | characters per chunk |
| `ChunkOverlap` | `50` | overlap ระหว่าง chunk |
| `EmbedRetryAttempts` | `3` | retry สำหรับ embedding fail |
| `EmbedRetryDelay` | `500ms` | delay ระหว่าง retry |

---

## 10. การ Integrate เข้าโปรเจกต์

### Input Format

Source name ต้องอยู่ในรูปแบบ: `"kbId:filename.ext:pageNumber"`

```go
sources := []am.KnowledgeSource{
    {
        SourceType:     am.AssetTypePage,
        SourceName:     "kb001:report.pdf:1",
        SourceContents: &page1Content,
    },
    {
        SourceType:     am.AssetTypePage,
        SourceName:     "kb001:report.pdf:2",
        SourceContents: &page2Content,
    },
}

result, err := indexer.Index(ctx, "kb001", sources, nil, nil, nil)
```

### Retrieve

```go
// ดึง seed assets ก่อน (document-level anchors)
seedAssets := indexer.GetRetrievalSeedAssets(ctx, query, kbIds, nil, nil)

// Retrieve
result, err := indexer.Retrieve(ctx, query, kbIds, seedAssets, nil)

// result.RetrievedAssets ประกอบด้วย:
// - entities ที่ rank สูงสุดจาก PPR (พร้อม score)
// - relations ที่เชื่อมต่อกับ entities เหล่านั้น
// - chunks ที่ entities/relations อ้างถึง
```

### ตัวอย่าง Full Flow (จาก example/)

```go
// 1. โหลด config
cfg, _ := LoadConfig("app")

// 2. สร้าง indexer
indexer, _ := InitializeFastGraphIndexerFromConfig(cfg)

// 3. สร้าง service layer
service := NewFastGraphService(extractors, *indexer)

// 4. Index ไฟล์
result, _ := service.Index(ctx, sourceFile, "kb001", nil, nil, nil)

// 5. Retrieve
retrieveResult, _ := service.Retrieve(ctx, "ใครเป็น CEO ของ บริษัท ก.?", []string{"kb001"}, nil)

// 6. ใช้ retrieveResult.RetrievedAssets เป็น context ให้ LLM ตอบคำถาม
```

### LLM Interface ที่ต้อง Implement

```go
type LLM interface {
    // ต้องรองรับ Tool Calling (Structured Output)
    Generate(
        ctx context.Context,
        messages []openai.ChatCompletionMessage,
        tools []openai.ChatCompletionToolParam,
        toolChoice *openai.ChatCompletionToolChoiceOptionUnionParam,
    ) (*openai.ChatCompletionMessage, error)
}
```

ระบบใช้ OpenAI-compatible API (รองรับ model ใดก็ได้ที่มี tool calling)

---

## ลำดับการอ่าน Code

สำหรับผู้ที่ต้องการเข้าใจ codebase แนะนำให้อ่านตามลำดับนี้:

1. **`models/graph.go`** — ทำความเข้าใจ data structures ก่อน
2. **`utils/const.go`** — ดู default values ทั้งหมด
3. **`graph/interface.go`** → **`graph/extractor.go`** — เข้าใจ LLM extraction
4. **`rag/options.go`** — เข้าใจ configuration system
5. **`rag/graph_rag.go`** ฟังก์ชัน `Index()` — อ่าน indexing pipeline
6. **`rag/graph_rag.go`** ฟังก์ชัน `getTransitionMatrixForKbId()` + `getSeedVector()` + `ppr()` — หัวใจของ algorithm
7. **`rag/graph_rag.go`** ฟังก์ชัน `Retrieve()` — อ่าน retrieval pipeline
8. **`indexer.go`** — Public API wrapper
9. **`example/`** — ดูการใช้งานจริง

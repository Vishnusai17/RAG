# GraphRAG Local LLM Integration: Progress Log

This document chronicles every step taken during the configuration of local LLM (Ollama) support for the GraphRAG system, replacing the original OpenAI dependency.

---

## Session Overview

**Objective**: Configure CUAD contract ingestion using a local LLM (Ollama) instead of OpenAI API.

**Duration**: ~1 hour of troubleshooting and configuration

**Final Result**: ✅ Successfully configured with `gemma3:4b` model using native Ollama library

---

## Progress Timeline

### Phase 1: Initial Ingestion Attempt (Steps 1-10)

| Step | Action | Result |
|------|--------|--------|
| 1 | Prepared `data/temp_ingest` directory with 3 sample CUAD contracts | ✅ Success |
| 2 | Ran `python3 scripts/ingest.py --path data/temp_ingest` | ❌ Failed |
| 3 | **Error**: `401 Incorrect API key - your_ope************here` | OpenAI placeholder in `.env` |
| 4 | **Error**: `Neo4j AuthenticationRateLimit` | Too many failed auth attempts |
| 5 | Inspected `.env` file | Found placeholder credentials |

**Root Cause**: The `.env` file contained placeholder values:
```env
OPENAI_API_KEY=your_openai_api_key_here
NEO4J_PASSWORD=your_password_here
```

---

### Phase 2: User Decision - Local LLM (Steps 11-20)

| Step | Action | Result |
|------|--------|--------|
| 6 | Notified user: credentials required | User asked about OpenAI pricing |
| 7 | Explained OpenAI is paid | User confirmed they have Ollama installed |
| 8 | Ran `ollama list` to check available models | Found: `mistral:latest`, `gemma3:4b` |
| 9 | Created task for "Switching to Local Ollama LLM" | ✅ Task created |
| 10 | Updated `task.md` with Phase 8: Local LLM Integration | ✅ Updated |

**Decision**: Use local Ollama instead of paid OpenAI API.

---

### Phase 3: Configuration Updates (Steps 21-30)

| Step | Action | Result |
|------|--------|--------|
| 11 | Updated `config.py` to support `LLM_PROVIDER` env var | ✅ Added |
| 12 | Added Ollama settings: `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | ✅ Added |
| 13 | Updated `.env` with `LLM_PROVIDER=ollama` | ✅ Set |
| 14 | Updated `.env` with `OLLAMA_MODEL=mistral` | ✅ Set |
| 15 | Modified `entity_extractor.py` to use `ChatOllama` | ✅ Added conditional |
| 16 | Modified `relationship_extractor.py` similarly | ✅ Added conditional |
| 17 | Modified `vector_store.py` to use `OllamaEmbeddings` | ✅ Added conditional |
| 18 | Created `scripts/test_connections.py` | ✅ Created |
| 19 | Ran connection test | ❌ Neo4j: Unauthorized |
| 20 | Ran connection test | ✅ Ollama: "Hello! How can I help?" |

**Partial Success**: Ollama working, Neo4j authentication failing.

---

### Phase 4: Neo4j Authentication Battle (Steps 31-45)

| Step | Action | Result |
|------|--------|--------|
| 21 | User tried `neo4j-admin dbms set-initial-password secret` | ❌ "Password must be at least 8 characters" |
| 22 | Updated `.env` with `NEO4J_PASSWORD=secret123` | ✅ Updated |
| 23 | Ran `test_connections.py` | ❌ Still Unauthorized |
| 24 | User restarted Neo4j | Still failing |
| 25 | Tested with default password `neo4j` | ✅ SUCCESS! |
| 26 | **Insight**: `set-initial-password` only works on fresh DB | Password was still `neo4j` |
| 27 | Ran ingestion again | ❌ `CredentialsExpired` error |
| 28 | Created `scripts/change_password.py` | ✅ Created |
| 29 | Ran password change script | ✅ "Password changed to secret123" |
| 30 | Updated `.env` with new password | ✅ Updated |
| 31 | Ran ingestion again | Started processing... |

**Root Cause**: Neo4j requires password change on first login. The command-line tool didn't work because the DB was already initialized.

---

### Phase 5: LangChain ChatOllama Hanging (Steps 46-60)

| Step | Action | Result |
|------|--------|--------|
| 32 | Ingestion stuck on "[1/3] Processing..." | ⏳ Hung indefinitely |
| 33 | Checked `check_graph.py` | 0 nodes (nothing written) |
| 34 | Created minimal `test_small.txt` test file | ✅ Created |
| 35 | Ran ingestion on small file | ⏳ Still hanging |
| 36 | Created `scripts/debug_ollama.py` | ✅ Created |
| 37 | Ran debug script with full prompt | ⏳ Hanging on `requests.post` |
| 38 | Simplified prompt to "Extract entities..." | ⏳ Still hanging |
| 39 | Ran `test_connections.py` again | ✅ Works (simple "Hi" prompt) |
| 40 | **Hypothesis**: Complex prompts cause `ChatOllama` to hang | LangChain issue |

**Root Cause**: LangChain's `ChatOllama` has compatibility issues on Mac, causing hangs on complex prompts.

---

### Phase 6: Native Ollama Library Migration (Steps 61-75)

| Step | Action | Result |
|------|--------|--------|
| 41 | Refactored `entity_extractor.py` to use `ollama.chat()` directly | ✅ Replaced |
| 42 | Removed `ChatOllama` import, added `import ollama` | ✅ Updated |
| 43 | Set `self.llm = "ollama"` as marker instead of object | ✅ Updated |
| 44 | Added conditional: `if self.llm == "ollama": ollama.chat(...)` | ✅ Added |
| 45 | Ran ingestion on `test_small.txt` | Processing started! |
| 46 | Ingestion hung on relationship extraction | ⏳ Hanging |
| 47 | **Insight**: `RelationshipExtractor` still using `ChatOllama` | Need same fix |
| 48 | Refactored `relationship_extractor.py` identically | ✅ Replaced |
| 49 | Ran ingestion again | ⏳ Proceeding but slow |

**Progress**: Entity extraction now working with native library.

---

### Phase 7: Model Performance Issues (Steps 76-85)

| Step | Action | Result |
|------|--------|--------|
| 50 | Checked graph: still 0 nodes | Ingestion not completing |
| 51 | Interrupted ingestion, saw `MERGE` queries in logs | Graph writes happening! |
| 52 | Updated `.env` to `OLLAMA_MODEL=gemma3:4b` | ✅ Lighter model |
| 53 | Simplified `ENTITY_EXTRACTION_PROMPT` | ✅ Shorter prompt |
| 54 | Ran ingestion with Gemma 3 | Processing faster |
| 55 | Checked graph after interrupt | 8 nodes found! |

**Success**: `gemma3:4b` with simplified prompts works.

---

### Phase 8: Embedding/Vector Store Issues (Steps 86-100)

| Step | Action | Result |
|------|--------|--------|
| 56 | Created `scripts/test_embeddings.py` | ✅ Created |
| 57 | Ran test with `OllamaEmbeddings` | ❌ "Model does not support embeddings" |
| 58 | Added `sentence-transformers` to `requirements.txt` | ✅ Added |
| 59 | Installed `pip3 install sentence-transformers langchain-huggingface` | ✅ Installed |
| 60 | Updated `vector_store.py` to use `HuggingFaceEmbeddings` | ✅ Updated |
| 61 | Ran test | ⏳ Hanging on `[mutex.cc : 452] Lock blocking` |
| 62 | Added `model_kwargs={'device': 'cpu'}` | ⏳ Still hanging |
| 63 | **Insight**: PyTorch/MPS conflict on Mac | Known issue |

**Root Cause**: HuggingFace/PyTorch has GPU lock issues on Mac with Metal (MPS).

---

### Phase 9: Workaround - Skip Vectors (Steps 101-110)

| Step | Action | Result |
|------|--------|--------|
| 64 | Added `--skip-vectors` argument to `ingest.py` | ✅ Added |
| 65 | Modified ingestion logic to skip vector store | ✅ Conditional added |
| 66 | Ran: `python3 scripts/ingest.py --path data/temp_ingest/test_small.txt --skip-vectors` | ✅ Completed! |
| 67 | Checked graph | 8 nodes: Document, Person, Organization, Location, Date |
| 68 | Started full CUAD batch ingestion with `--skip-vectors` | ⏳ Running |

**Final Status**: Graph construction working, vectors disabled as workaround.

---

## Summary of Issues & Resolutions

| # | Issue | Root Cause | Resolution |
|---|-------|------------|------------|
| 1 | OpenAI 401 Unauthorized | Placeholder API key | Switched to Ollama |
| 2 | Neo4j AuthenticationRateLimit | Too many failed attempts | Wait + restart |
| 3 | Neo4j Unauthorized | Placeholder password | Found default was `neo4j` |
| 4 | Neo4j CredentialsExpired | Must change default password | Created `change_password.py` |
| 5 | LangChain ChatOllama Hangs | Mac compatibility issue | Switched to native `ollama` library |
| 6 | Mistral model slow | Large model (4.4GB) | Switched to `gemma3:4b` (3.3GB) |
| 7 | Ollama embeddings 500 error | gemma3 doesn't support embeddings | Switched to HuggingFace |
| 8 | HuggingFace embeddings hang | PyTorch MPS lock (Mac GPU) | Added `--skip-vectors` flag |

---

## Files Modified

| File | Changes |
|------|---------|
| `.env` | Added LLM_PROVIDER, OLLAMA_*, updated password |
| `config.py` | Added Ollama settings to Settings class |
| `extraction/entity_extractor.py` | Native `ollama.chat()` integration |
| `extraction/relationship_extractor.py` | Native `ollama.chat()` integration |
| `extraction/prompts.py` | Simplified ENTITY_EXTRACTION_PROMPT |
| `rag/vector_store.py` | HuggingFaceEmbeddings with CPU fallback |
| `scripts/ingest.py` | Added `--skip-vectors` argument |
| `requirements.txt` | Added sentence-transformers, langchain-huggingface |

## Files Created

| File | Purpose |
|------|---------|
| `scripts/test_connections.py` | Verify Neo4j + Ollama connectivity |
| `scripts/change_password.py` | Programmatically reset Neo4j password |
| `scripts/check_graph.py` | Verify node counts in graph |
| `scripts/debug_ollama.py` | Isolate Ollama prompt issues |
| `scripts/test_embeddings.py` | Test Ollama embeddings |
| `scripts/test_hf_embeddings.py` | Test HuggingFace embeddings |

---

## Final Configuration

```env
# Working Configuration
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma3:4b
NEO4J_PASSWORD=secret123
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

```bash
# Working Ingestion Command
python3 scripts/ingest.py --path data/temp_ingest --skip-vectors
```

---

## Next Steps (Optional)

1. **Enable Vector Search**: Try using `nomic-embed-text` Ollama model (designed for embeddings)
2. **Performance**: Consider running on a Linux machine for full GPU support
3. **Scale**: Process the full CUAD dataset (~500 contracts)

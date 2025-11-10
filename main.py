# """
# rag_faiss_local.py

# - Ingest local text sources into a FAISS IndexIDMap with sentence-transformers embeddings.
# - Query the index for top-K contexts for RAG.
# - Generate replies via llama-cpp-python (preferred) or Transformers fallback (if llama-cpp not available).
# - Metadata mapping stored in JSON; index saved to disk.

# Usage:
#   python rag_faiss_local.py ingest ./data   # create/update index from files in ./data
#   python rag_faiss_local.py chat            # interactive chat loop (uses built index)
# """

# import os
# import sys
# import json
# import time
# import math
# import glob
# import uuid
# import argparse
# from typing import List, Dict, Any
# from tqdm import tqdm

# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Try to import llama-cpp; if unavailable, we'll fallback to transformers
# LLAMA_AVAILABLE = False
# TRANSFORMERS_AVAILABLE = False
# try:
#     from llama_cpp import Llama
#     LLAMA_AVAILABLE = True
# except Exception:
#     LLAMA_AVAILABLE = False

# try:
#     from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#     TRANSFORMERS_AVAILABLE = True
# except Exception:
#     TRANSFORMERS_AVAILABLE = False

# # FAISS import
# try:
#     import faiss
# except Exception as e:
#     raise RuntimeError("faiss is required. Install faiss-cpu (pip) or faiss via conda on Windows.") from e

# # -------------------- CONFIG --------------------
# EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# EMBED_DIM = 384  # matches the model above
# FAISS_INDEX_PATH = "faiss_index.ivf"   # file to save FAISS index
# META_PATH = "meta_store.json"          # maps int_id -> metadata dict

# # LLM config (edit)
# LLAMA_MODEL_PATH = "\\tholospg.itserv.scss.tcd.ie\Pgrad\mkhan5\My Documents\GitHub\Visual-Novel-With-Custom-Character-Interactions-Using-LLM-RAG\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # set if you have a ggml model for llama-cpp
# LLAMA_CTX = 2048
# LLAMA_TEMPERATURE = 0.7
# LLAMA_MAX_TOKENS = 256

# # Transformers fallback (if no llama-cpp)
# HF_MODEL_ID = "gpt2"  # replace with a local bigger model if available (gpt2 is small, demo-only)
# HF_MAX_NEW_TOKENS = 256
# HF_TEMPERATURE = 0.7

# # RAG params
# TOP_K = 6
# CONTEXT_TOKEN_BUDGET = 1500  # approx tokens reserved for contexts (trim if exceed model ctx)
# # ------------------------------------------------

# # Load embedding model once
# print("Loading embedding model:", EMBED_MODEL_NAME)
# embedder = SentenceTransformer(EMBED_MODEL_NAME)


# # -------------------- Utilities --------------------
# def save_json(path: str, obj: Any):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, ensure_ascii=False, indent=2)


# def load_json(path: str) -> Any:
#     if not os.path.exists(path):
#         return {}
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# # Simple chunker: split text into chunks by paragraphs, then sliding-window merge to target size
# def chunk_text(text: str, max_chunk_chars: int = 800) -> List[str]:
#     paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
#     chunks: List[str] = []
#     current = ""
#     for p in paragraphs:
#         if len(current) + len(p) + 1 <= max_chunk_chars:
#             current = (current + "\n\n" + p).strip() if current else p
#         else:
#             if current:
#                 chunks.append(current)
#             if len(p) <= max_chunk_chars:
#                 current = p
#             else:
#                 # fallback: break by sentences if paragraph too long
#                 sentences = p.split(". ")
#                 cur2 = ""
#                 for s in sentences:
#                     if len(cur2) + len(s) + 2 <= max_chunk_chars:
#                         cur2 = (cur2 + ". " + s).strip() if cur2 else s
#                     else:
#                         if cur2:
#                             chunks.append(cur2.strip() + ".")
#                         cur2 = s
#                 if cur2:
#                     current = cur2.strip()
#                 else:
#                     current = ""
#     if current:
#         chunks.append(current)
#     return chunks


# # -------------------- FAISS INDEX helpers --------------------
# def create_faiss_index(emb_dim: int):
#     # Using IndexFlatIP + IndexIDMap for cosine (use normalized vectors)
#     index = faiss.IndexFlatIP(emb_dim)
#     id_index = faiss.IndexIDMap(index)
#     return id_index


# def save_faiss(index, path: str):
#     faiss.write_index(index, path)
#     print("Saved faiss index to", path)


# def load_faiss(path: str):
#     if not os.path.exists(path):
#         return create_faiss_index(EMBED_DIM)
#     idx = faiss.read_index(path)
#     # ensure it's wrapped in IDMap (it should be if you saved it that way)
#     try:
#         _ = idx.id_map
#         return idx
#     except Exception:
#         # wrap
#         return faiss.IndexIDMap(idx)


# # -------------------- Ingest pipeline --------------------
# def ingest_folder(data_folder: str, index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH,
#                   batch_size: int = 64):
#     """
#     Walks data_folder for .txt, .jsonl, .json, .csv files and ingests textual items into FAISS.
#     - txt: entire file becomes one document
#     - jsonl/json: expects objects with 'id' and 'text' (or will generate an id)
#     - csv: first column is treated as text if header unknown
#     """
#     # load existing meta and index
#     meta = load_json(meta_path)  # maps str(int_id) -> metadata
#     index = load_faiss(index_path)

#     # build a set of used_int_ids to avoid collisions
#     used_int_ids = set(int(k) for k in meta.keys()) if meta else set()
#     next_int_id = max(used_int_ids) + 1 if used_int_ids else 1

#     # collect new documents as tuples (int_id, metadata, text)
#     new_docs = []

#     # helper to add doc
#     def add_doc(text, source_name, orig_id=None, metadata_extra=None):
#         nonlocal next_int_id
#         if not text or not text.strip():
#             return
#         chunks = chunk_text(text)
#         for i, c in enumerate(chunks):
#             int_id = next_int_id
#             next_int_id += 1
#             doc_id = orig_id or str(uuid.uuid4())
#             metadata = {
#                 "doc_id": doc_id,
#                 "source": source_name,
#                 "chunk_index": i,
#                 "text": c,
#             }
#             if metadata_extra:
#                 metadata.update(metadata_extra)
#             new_docs.append((int_id, metadata, c))

#     # walk files
#     patterns = ["*.txt", "*.jsonl", "*.json", "*.csv"]
#     file_list = []
#     for p in patterns:
#         file_list.extend(glob.glob(os.path.join(data_folder, p)))
#     print(f"Found {len(file_list)} files to ingest in {data_folder}")

#     for filepath in file_list:
#         ext = os.path.splitext(filepath)[1].lower()
#         name = os.path.basename(filepath)
#         try:
#             if ext == ".txt":
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     text = f.read()
#                 add_doc(text, source_name=name)
#             elif ext == ".jsonl":
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     for line in f:
#                         obj = json.loads(line)
#                         txt = obj.get("text") or obj.get("dialogue") or obj.get("content")
#                         add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
#             elif ext == ".json":
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#                 # if it's a list of objects
#                 if isinstance(data, list):
#                     for obj in data:
#                         txt = obj.get("text") or obj.get("content")
#                         add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
#                 elif isinstance(data, dict):
#                     # try to find a key that is a list of items
#                     if "items" in data and isinstance(data["items"], list):
#                         for obj in data["items"]:
#                             txt = obj.get("text") or obj.get("content")
#                             add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
#                     else:
#                         # treat file as single doc
#                         txt = data.get("text") or json.dumps(data)
#                         add_doc(txt, source_name=name, orig_id=data.get("id"), metadata_extra=data)
#             elif ext == ".csv":
#                 import csv
#                 with open(filepath, newline='', encoding='utf-8') as csvfile:
#                     reader = csv.reader(csvfile)
#                     for row in reader:
#                         if not row:
#                             continue
#                         txt = row[0]
#                         add_doc(txt, source_name=name)
#             else:
#                 # ignore unknown
#                 pass
#         except Exception as e:
#             print("Failed to ingest", filepath, e)

#     if not new_docs:
#         print("No new docs found to ingest.")
#         return

#     # compute embeddings in batches
#     texts = [d[2] for d in new_docs]
#     int_ids = [d[0] for d in new_docs]
#     metas = {str(d[0]): d[1] for d in new_docs}

#     # batch encode
#     embeddings = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
#         batch = texts[i:i + batch_size]
#         emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
#         embeddings.append(emb)
#     embeddings = np.vstack(embeddings).astype('float32')

#     # normalize for cosine similarity with inner-product index
#     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#     norms[norms == 0] = 1e-9
#     embeddings = embeddings / norms

#     # add to faiss index using add_with_ids
#     id_array = np.array(int_ids, dtype=np.int64)
#     index.add_with_ids(embeddings, id_array)
#     print(f"Added {len(int_ids)} vectors to FAISS index.")

#     # merge metadata and save
#     meta.update(metas)
#     save_json(meta_path, meta)
#     save_faiss(index, index_path)
#     print("Ingest complete.")


# # -------------------- Retrieval & prompt building --------------------
# def retrieve(query: str, top_k: int = TOP_K, index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH):
#     index = load_faiss(index_path)
#     meta = load_json(meta_path)
#     # encode
#     q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype('float32')
#     q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
#     D, I = index.search(np.expand_dims(q_emb, axis=0), top_k)
#     results = []
#     for dist, iid in zip(D[0], I[0]):
#         if int(iid) == -1:
#             continue
#         key = str(int(iid))
#         if key in meta:
#             entry = meta[key].copy()
#             entry["_score"] = float(dist)
#             results.append(entry)
#     return results


# def build_prompt(character_name: str, persona_notes: str, contexts: List[Dict], short_memory: str, player_input: str):
#     # contexts are ordered most relevant first
#     ctx_texts = []
#     total_chars = 0
#     for c in contexts:
#         s = f"[{c.get('source','unknown')}|chunk{c.get('chunk_index')}] {c.get('text')}"
#         ctx_texts.append(s)
#         total_chars += len(s)
#     contexts_joined = "\n\n---\n\n".join(ctx_texts)

#     prompt = f"""System: You are {character_name}. The following describes your voice, personality, and rules. Respond in-character, concisely, and do not invent facts beyond the Contexts or your Short Memory. If the player asks about something you don't know, politely deflect.

# Persona:
# {persona_notes}

# Contexts (most relevant first):
# {contexts_joined}

# Short memory:
# {short_memory or "None"}

# Player: {player_input}
# {character_name}:"""
#     return prompt


# # -------------------- LLM wrappers --------------------
# class LlamaWrapper:
#     def __init__(self, model_path: str, n_ctx: int = LLAMA_CTX):
#         if not LLAMA_AVAILABLE:
#             raise RuntimeError("llama-cpp-python not available")
#         print("Loading llama-cpp model:", model_path)
#         self.llm = Llama(model_path=model_path, n_ctx=n_ctx)

#     def generate(self, prompt: str, max_tokens: int = LLAMA_MAX_TOKENS, temperature: float = LLAMA_TEMPERATURE):
#         # Use stop tokens to prevent it from continuing the dialogue
#         resp = self.llm.create_chat_completion(
#             prompt=prompt,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             stop=["Player:", "\nPlayer:", "\nSystem:"]
#         )
#         if isinstance(resp, dict):
#             choices = resp.get("choices")
#             if choices and isinstance(choices, list):
#                 txt = choices[0].get("text") or choices[0].get("message", {}).get("content", "")
#                 # Trim anything after it starts echoing the player
#                 txt = txt.split("Player:")[0].strip()
#                 return txt
#         return str(resp).split("Player:")[0].strip()

# class HFWrapper:
#     def __init__(self, model_id: str = HF_MODEL_ID):
#         if not TRANSFORMERS_AVAILABLE:
#             raise RuntimeError("Transformers not available")
#         print("Loading HF model:", model_id)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
#         self.pipe = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             device_map="auto"
#         )

#     def generate(self, prompt: str, max_tokens: int = HF_MAX_NEW_TOKENS, temperature: float = HF_TEMPERATURE):
#         stop_seq = ["Player:", "\nPlayer:", "\nSystem:"]
#         out = self.pipe(
#             prompt,
#             max_new_tokens=max_tokens,
#             do_sample=True,
#             temperature=temperature,
#             return_full_text=False
#         )[0]["generated_text"]
#         # Stop if it starts to include the Player's turn
#         for stop in stop_seq:
#             if stop in out:
#                 out = out.split(stop)[0]
#                 break
#         return out.strip()


# # -------------------- Respond function --------------------
# def respond(player_input: str, character_name: str, persona_notes: str, short_memory: str,
#             llm_wrapper, index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH,
#             top_k: int = TOP_K):
#     # 1) retrieve contexts
#     contexts = retrieve(player_input, top_k, index_path, meta_path)
#     # 2) build prompt (we rely on the wrapper to obey persona instructions)
#     prompt = build_prompt(character_name, persona_notes, contexts, short_memory, player_input)

#     # Basic prompt token budget enforcement (very rough: char -> token ~ 4 chars per token)
#     approx_tokens = len(prompt) / 4
#     if approx_tokens + LLAMA_MAX_TOKENS > LLAMA_CTX:
#         # trimming contexts: keep first N contexts until under budget
#         trimmed = []
#         tokens_used = len(f"System: You are {character_name}.") / 4
#         for c in contexts:
#             s = c.get("text", "")
#             t = len(s) / 4
#             if tokens_used + t + LLAMA_MAX_TOKENS < LLAMA_CTX:
#                 trimmed.append(c)
#                 tokens_used += t
#             else:
#                 break
#         contexts = trimmed
#         prompt = build_prompt(character_name, persona_notes, contexts, short_memory, player_input)

#     # 3) generate
#     reply = llm_wrapper.generate(prompt)
#     return reply, contexts, prompt


# # -------------------- CLI --------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode", choices=["ingest", "chat"], help="ingest or chat")
#     parser.add_argument("path", nargs="?", help="path to data folder for ingest")
#     args = parser.parse_args()

#     if args.mode == "ingest":
#         if not args.path:
#             print("Provide a folder path to ingest, e.g. python rag_faiss_local.py ingest ./data")
#             sys.exit(1)
#         ingest_folder(args.path)
#         print("Ingest finished.")
#         sys.exit(0)

#     # chat mode
#     # initialize LLM wrapper
#     llm = None
#     if LLAMA_AVAILABLE and os.path.exists(LLAMA_MODEL_PATH):
#         try:
#             llm = LlamaWrapper(LLAMA_MODEL_PATH)
#             print("Using llama-cpp backend.")
#         except Exception as e:
#             print("Failed to init llama-cpp:", e)
#             llm = None

#     if llm is None and TRANSFORMERS_AVAILABLE:
#         try:
#             llm = HFWrapper(HF_MODEL_ID)
#             print("Using Transformers backend (fallback).")
#         except Exception as e:
#             print("Failed to init Transformers model:", e)
#             llm = None

#     if llm is None:
#         print("No LLM backend available. Install llama-cpp-python and/or transformers.")
#         sys.exit(1)

#     # basic persona and memory for testing
#     character_name = "Eri"
#     persona_notes = (
#         "Soft-spoken, quick-witted, composed but with a dry humor. "
#         "Speaks concisely, avoids giving out private secrets, and deflects questions she doesn't know."
#     )
#     short_memory = "She likes jasmine tea and keeps a sketchbook in the café."

#     print("Interactive chat. Type 'exit' to quit.")
#     while True:
#         q = input("Player: ").strip()
#         if not q:
#             continue
#         if q.lower() in ("quit", "exit"):
#             break
#         reply, contexts, prompt = respond(q, character_name, persona_notes, short_memory, llm)
#         print(f"{character_name}: {reply}\n")
#         # for debugging, you can optionally print contexts:
#         # print("Contexts used:", [c['source'] for c in contexts])

# if __name__ == "__main__":
#     main()

# """
# rag_faiss_local.py

# - Ingest local text sources into a FAISS IndexIDMap with sentence-transformers embeddings.
# - Query the index for top-K contexts for RAG.
# - Generate replies via llama-cpp-python (preferred) or Transformers fallback (if llama-cpp not available).
# - Metadata mapping stored in JSON; index saved to disk.
# - **NEW**: Memory system that extracts and stores key moments from conversations

# Usage:
#   python rag_faiss_local.py ingest ./data   # create/update index from files in ./data
#   python rag_faiss_local.py chat            # interactive chat loop (uses built index)
# """

# import os
# import sys
# import json
# import time
# import math
# import glob
# import uuid
# import argparse
# from typing import List, Dict, Any
# from datetime import datetime
# from tqdm import tqdm

# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Try to import llama-cpp; if unavailable, we'll fallback to transformers
# LLAMA_AVAILABLE = False
# TRANSFORMERS_AVAILABLE = False
# try:
#     from llama_cpp import Llama
#     LLAMA_AVAILABLE = True
# except Exception:
#     LLAMA_AVAILABLE = False

# try:
#     from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#     TRANSFORMERS_AVAILABLE = True
# except Exception:
#     TRANSFORMERS_AVAILABLE = False

# # FAISS import
# try:
#     import faiss
# except Exception as e:
#     raise RuntimeError("faiss is required. Install faiss-cpu (pip) or faiss via conda on Windows.") from e

# # -------------------- CONFIG --------------------
# EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# EMBED_DIM = 384  # matches the model above
# FAISS_INDEX_PATH = "faiss_index.ivf"   # file to save FAISS index
# META_PATH = "meta_store.json"          # maps int_id -> metadata dict
# MEMORY_PATH = "conversation_memory.json"  # stores key memories from conversations

# # LLM config (edit)
# LLAMA_MODEL_PATH = "\\\\tholospg.itserv.scss.tcd.ie\\Pgrad\\mkhan5\\My Documents\\GitHub\\Visual-Novel-With-Custom-Character-Interactions-Using-LLM-RAG\\models\\mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # set if you have a ggml model for llama-cpp
# LLAMA_CTX = 2048
# LLAMA_TEMPERATURE = 0.7
# LLAMA_MAX_TOKENS = 256

# # Transformers fallback (if no llama-cpp)
# HF_MODEL_ID = "gpt2"  # replace with a local bigger model if available (gpt2 is small, demo-only)
# HF_MAX_NEW_TOKENS = 256
# HF_TEMPERATURE = 0.7

# # RAG params
# TOP_K = 6
# CONTEXT_TOKEN_BUDGET = 1500  # approx tokens reserved for contexts (trim if exceed model ctx)
# MEMORY_TOP_K = 3  # number of relevant memories to include in prompts
# # ------------------------------------------------

# # Load embedding model once
# print("Loading embedding model:", EMBED_MODEL_NAME)
# embedder = SentenceTransformer(EMBED_MODEL_NAME)


# # -------------------- Utilities --------------------
# def save_json(path: str, obj: Any):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, ensure_ascii=False, indent=2)


# def load_json(path: str) -> Any:
#     if not os.path.exists(path):
#         return {}
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# # Simple chunker: split text into chunks by paragraphs, then sliding-window merge to target size
# def chunk_text(text: str, max_chunk_chars: int = 800) -> List[str]:
#     paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
#     chunks: List[str] = []
#     current = ""
#     for p in paragraphs:
#         if len(current) + len(p) + 1 <= max_chunk_chars:
#             current = (current + "\n\n" + p).strip() if current else p
#         else:
#             if current:
#                 chunks.append(current)
#             if len(p) <= max_chunk_chars:
#                 current = p
#             else:
#                 # fallback: break by sentences if paragraph too long
#                 sentences = p.split(". ")
#                 cur2 = ""
#                 for s in sentences:
#                     if len(cur2) + len(s) + 2 <= max_chunk_chars:
#                         cur2 = (cur2 + ". " + s).strip() if cur2 else s
#                     else:
#                         if cur2:
#                             chunks.append(cur2.strip() + ".")
#                         cur2 = s
#                 if cur2:
#                     current = cur2.strip()
#                 else:
#                     current = ""
#     if current:
#         chunks.append(current)
#     return chunks


# # -------------------- FAISS INDEX helpers --------------------
# def create_faiss_index(emb_dim: int):
#     # Using IndexFlatIP + IndexIDMap for cosine (use normalized vectors)
#     index = faiss.IndexFlatIP(emb_dim)
#     id_index = faiss.IndexIDMap(index)
#     return id_index


# def save_faiss(index, path: str):
#     faiss.write_index(index, path)
#     print("Saved faiss index to", path)


# def load_faiss(path: str):
#     if not os.path.exists(path):
#         return create_faiss_index(EMBED_DIM)
#     idx = faiss.read_index(path)
#     # ensure it's wrapped in IDMap (it should be if you saved it that way)
#     try:
#         _ = idx.id_map
#         return idx
#     except Exception:
#         # wrap
#         return faiss.IndexIDMap(idx)


# # -------------------- Memory System --------------------
# class ConversationMemory:
#     """Stores and retrieves key moments from conversations"""
    
#     def __init__(self, memory_path: str = MEMORY_PATH):
#         self.memory_path = memory_path
#         self.memories = self._load_memories()
#         self.memory_embeddings = None
#         self.memory_index = None
#         self._build_memory_index()
    
#     def _load_memories(self) -> List[Dict]:
#         """Load memories from disk"""
#         if os.path.exists(self.memory_path):
#             with open(self.memory_path, "r", encoding="utf-8") as f:
#                 return json.load(f)
#         return []
    
#     def _save_memories(self):
#         """Save memories to disk"""
#         save_json(self.memory_path, self.memories)
    
#     def _build_memory_index(self):
#         """Build FAISS index for semantic search over memories"""
#         if not self.memories:
#             return
        
#         texts = [m["memory_text"] for m in self.memories]
#         embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
#         embeddings = embeddings.astype('float32')
        
#         # Normalize for cosine similarity
#         norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#         norms[norms == 0] = 1e-9
#         embeddings = embeddings / norms
        
#         self.memory_embeddings = embeddings
#         self.memory_index = faiss.IndexFlatIP(EMBED_DIM)
#         self.memory_index.add(embeddings)
    
#     def extract_memory(self, player_input: str, character_response: str, llm_wrapper) -> str:
#         """Use LLM to extract key information worth remembering"""
#         extraction_prompt = f"""Analyze this conversation exchange and extract any key facts, preferences, or important moments worth remembering. Focus on:
# - Personal preferences (favorite colors, foods, hobbies)
# - Important facts about the player
# - Significant events or revelations
# - Emotional moments or decisions

# If there's nothing particularly memorable, respond with "NONE".
# If there is something memorable, write a single concise sentence (under 20 words) summarizing what to remember.

# Player: {player_input}
# Character: {character_response}

# Memory to extract (or NONE):"""
        
#         try:
#             if isinstance(llm_wrapper, LlamaWrapper):
#                 memory = llm_wrapper.llm.create(
#                     prompt=extraction_prompt,
#                     max_tokens=50,
#                     temperature=0.3,
#                     stop=["\n"]
#                 )
#                 if isinstance(memory, dict):
#                     memory_text = memory.get("choices", [{}])[0].get("text", "").strip()
#                 else:
#                     memory_text = str(memory).strip()
#             else:
#                 memory_text = llm_wrapper.generate(extraction_prompt, max_tokens=50, temperature=0.3)
            
#             memory_text = memory_text.replace("Memory to extract:", "").strip()
            
#             if memory_text and memory_text.upper() != "NONE" and len(memory_text) > 5:
#                 return memory_text
#         except Exception as e:
#             print(f"Memory extraction error: {e}")
        
#         return None
    
#     def add_memory(self, memory_text: str, player_input: str, character_response: str):
#         """Add a new memory to the system"""
#         memory = {
#             "id": str(uuid.uuid4()),
#             "memory_text": memory_text,
#             "timestamp": datetime.now().isoformat(),
#             "context": {
#                 "player_input": player_input,
#                 "character_response": character_response
#             }
#         }
#         self.memories.append(memory)
#         self._save_memories()
#         self._build_memory_index()  # Rebuild index with new memory
#         print(f"[Memory saved: {memory_text}]")
    
#     def retrieve_relevant_memories(self, query: str, top_k: int = MEMORY_TOP_K) -> List[Dict]:
#         """Retrieve most relevant memories for current context"""
#         if not self.memories or self.memory_index is None:
#             return []
        
#         # Encode query
#         q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype('float32')
#         q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        
#         # Search
#         k = min(top_k, len(self.memories))
#         D, I = self.memory_index.search(np.expand_dims(q_emb, axis=0), k)
        
#         results = []
#         for score, idx in zip(D[0], I[0]):
#             if idx >= 0 and idx < len(self.memories):
#                 memory = self.memories[idx].copy()
#                 memory["relevance_score"] = float(score)
#                 results.append(memory)
        
#         return results
    
#     def get_all_memories_summary(self) -> str:
#         """Get a summary of all memories for context"""
#         if not self.memories:
#             return "No memories yet."
        
#         recent = self.memories[-5:]  # Last 5 memories
#         summary = "\n".join([f"- {m['memory_text']}" for m in recent])
#         return summary


# # -------------------- Ingest pipeline --------------------
# def ingest_folder(data_folder: str, index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH,
#                   batch_size: int = 64):
#     """
#     Walks data_folder for .txt, .jsonl, .json, .csv files and ingests textual items into FAISS.
#     - txt: entire file becomes one document
#     - jsonl/json: expects objects with 'id' and 'text' (or will generate an id)
#     - csv: first column is treated as text if header unknown
#     """
#     # load existing meta and index
#     meta = load_json(meta_path)  # maps str(int_id) -> metadata
#     index = load_faiss(index_path)

#     # build a set of used_int_ids to avoid collisions
#     used_int_ids = set(int(k) for k in meta.keys()) if meta else set()
#     next_int_id = max(used_int_ids) + 1 if used_int_ids else 1

#     # collect new documents as tuples (int_id, metadata, text)
#     new_docs = []

#     # helper to add doc
#     def add_doc(text, source_name, orig_id=None, metadata_extra=None):
#         nonlocal next_int_id
#         if not text or not text.strip():
#             return
#         chunks = chunk_text(text)
#         for i, c in enumerate(chunks):
#             int_id = next_int_id
#             next_int_id += 1
#             doc_id = orig_id or str(uuid.uuid4())
#             metadata = {
#                 "doc_id": doc_id,
#                 "source": source_name,
#                 "chunk_index": i,
#                 "text": c,
#             }
#             if metadata_extra:
#                 metadata.update(metadata_extra)
#             new_docs.append((int_id, metadata, c))

#     # walk files
#     patterns = ["*.txt", "*.jsonl", "*.json", "*.csv"]
#     file_list = []
#     for p in patterns:
#         file_list.extend(glob.glob(os.path.join(data_folder, p)))
#     print(f"Found {len(file_list)} files to ingest in {data_folder}")

#     for filepath in file_list:
#         ext = os.path.splitext(filepath)[1].lower()
#         name = os.path.basename(filepath)
#         try:
#             if ext == ".txt":
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     text = f.read()
#                 add_doc(text, source_name=name)
#             elif ext == ".jsonl":
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     for line in f:
#                         obj = json.loads(line)
#                         txt = obj.get("text") or obj.get("dialogue") or obj.get("content")
#                         add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
#             elif ext == ".json":
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#                 # if it's a list of objects
#                 if isinstance(data, list):
#                     for obj in data:
#                         txt = obj.get("text") or obj.get("content")
#                         add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
#                 elif isinstance(data, dict):
#                     # try to find a key that is a list of items
#                     if "items" in data and isinstance(data["items"], list):
#                         for obj in data["items"]:
#                             txt = obj.get("text") or obj.get("content")
#                             add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
#                     else:
#                         # treat file as single doc
#                         txt = data.get("text") or json.dumps(data)
#                         add_doc(txt, source_name=name, orig_id=data.get("id"), metadata_extra=data)
#             elif ext == ".csv":
#                 import csv
#                 with open(filepath, newline='', encoding='utf-8') as csvfile:
#                     reader = csv.reader(csvfile)
#                     for row in reader:
#                         if not row:
#                             continue
#                         txt = row[0]
#                         add_doc(txt, source_name=name)
#             else:
#                 # ignore unknown
#                 pass
#         except Exception as e:
#             print("Failed to ingest", filepath, e)

#     if not new_docs:
#         print("No new docs found to ingest.")
#         return

#     # compute embeddings in batches
#     texts = [d[2] for d in new_docs]
#     int_ids = [d[0] for d in new_docs]
#     metas = {str(d[0]): d[1] for d in new_docs}

#     # batch encode
#     embeddings = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
#         batch = texts[i:i + batch_size]
#         emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
#         embeddings.append(emb)
#     embeddings = np.vstack(embeddings).astype('float32')

#     # normalize for cosine similarity with inner-product index
#     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#     norms[norms == 0] = 1e-9
#     embeddings = embeddings / norms

#     # add to faiss index using add_with_ids
#     id_array = np.array(int_ids, dtype=np.int64)
#     index.add_with_ids(embeddings, id_array)
#     print(f"Added {len(int_ids)} vectors to FAISS index.")

#     # merge metadata and save
#     meta.update(metas)
#     save_json(meta_path, meta)
#     save_faiss(index, index_path)
#     print("Ingest complete.")


# # -------------------- Retrieval & prompt building --------------------
# def retrieve(query: str, top_k: int = TOP_K, index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH):
#     index = load_faiss(index_path)
#     meta = load_json(meta_path)
#     # encode
#     q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype('float32')
#     q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
#     D, I = index.search(np.expand_dims(q_emb, axis=0), top_k)
#     results = []
#     for dist, iid in zip(D[0], I[0]):
#         if int(iid) == -1:
#             continue
#         key = str(int(iid))
#         if key in meta:
#             entry = meta[key].copy()
#             entry["_score"] = float(dist)
#             results.append(entry)
#     return results


# def build_prompt(character_name: str, persona_notes: str, contexts: List[Dict], 
#                 memories: List[Dict], short_memory: str, player_input: str):
#     # contexts are ordered most relevant first
#     ctx_texts = []
#     for c in contexts:
#         s = f"[{c.get('source','unknown')}|chunk{c.get('chunk_index')}] {c.get('text')}"
#         ctx_texts.append(s)
#     contexts_joined = "\n\n---\n\n".join(ctx_texts)
    
#     # Format memories
#     memory_section = ""
#     if memories:
#         memory_texts = [f"- {m['memory_text']}" for m in memories]
#         memory_section = f"\n\nKey Memories (things you remember about the player):\n" + "\n".join(memory_texts)

#     prompt = f"""System: You are {character_name}. The following describes your voice, personality, and rules. Respond in-character, concisely, and do not invent facts beyond the Contexts or your Memories. If the player asks about something you don't know, politely deflect.

# Persona:
# {persona_notes}

# Contexts (most relevant first):
# {contexts_joined}{memory_section}

# Short memory:
# {short_memory or "None"}

# Player: {player_input}
# {character_name}:"""
#     return prompt


# # -------------------- LLM wrappers --------------------
# class LlamaWrapper:
#     def __init__(self, model_path: str, n_ctx: int = LLAMA_CTX):
#         if not LLAMA_AVAILABLE:
#             raise RuntimeError("llama-cpp-python not available")
#         print("Loading llama-cpp model:", model_path)
#         self.llm = Llama(model_path=model_path, n_ctx=n_ctx)

#     def generate(self, prompt: str, max_tokens: int = LLAMA_MAX_TOKENS, temperature: float = LLAMA_TEMPERATURE):
#         # Use stop tokens to prevent it from continuing the dialogue
#         resp = self.llm.create_chat_completion(
#             prompt=prompt,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             stop=["Player:", "\nPlayer:", "\nSystem:"]
#         )
#         if isinstance(resp, dict):
#             choices = resp.get("choices")
#             if choices and isinstance(choices, list):
#                 txt = choices[0].get("text") or choices[0].get("message", {}).get("content", "")
#                 # Trim anything after it starts echoing the player
#                 txt = txt.split("Player:")[0].strip()
#                 return txt
#         return str(resp).split("Player:")[0].strip()

# class HFWrapper:
#     def __init__(self, model_id: str = HF_MODEL_ID):
#         if not TRANSFORMERS_AVAILABLE:
#             raise RuntimeError("Transformers not available")
#         print("Loading HF model:", model_id)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
#         self.pipe = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             device_map="auto"
#         )

#     def generate(self, prompt: str, max_tokens: int = HF_MAX_NEW_TOKENS, temperature: float = HF_TEMPERATURE):
#         stop_seq = ["Player:", "\nPlayer:", "\nSystem:"]
#         out = self.pipe(
#             prompt,
#             max_new_tokens=max_tokens,
#             do_sample=True,
#             temperature=temperature,
#             return_full_text=False
#         )[0]["generated_text"]
#         # Stop if it starts to include the Player's turn
#         for stop in stop_seq:
#             if stop in out:
#                 out = out.split(stop)[0]
#                 break
#         return out.strip()


# # -------------------- Respond function --------------------
# def respond(player_input: str, character_name: str, persona_notes: str, short_memory: str,
#             llm_wrapper, memory_system: ConversationMemory, 
#             index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH,
#             top_k: int = TOP_K):
#     # 1) retrieve contexts
#     contexts = retrieve(player_input, top_k, index_path, meta_path)
    
#     # 2) retrieve relevant memories
#     relevant_memories = memory_system.retrieve_relevant_memories(player_input)
    
#     # 3) build prompt
#     prompt = build_prompt(character_name, persona_notes, contexts, relevant_memories, 
#                          short_memory, player_input)

#     # Basic prompt token budget enforcement (very rough: char -> token ~ 4 chars per token)
#     approx_tokens = len(prompt) / 4
#     if approx_tokens + LLAMA_MAX_TOKENS > LLAMA_CTX:
#         # trimming contexts: keep first N contexts until under budget
#         trimmed = []
#         tokens_used = len(f"System: You are {character_name}.") / 4
#         for c in contexts:
#             s = c.get("text", "")
#             t = len(s) / 4
#             if tokens_used + t + LLAMA_MAX_TOKENS < LLAMA_CTX:
#                 trimmed.append(c)
#                 tokens_used += t
#             else:
#                 break
#         contexts = trimmed
#         prompt = build_prompt(character_name, persona_notes, contexts, relevant_memories,
#                             short_memory, player_input)

#     # 4) generate response
#     reply = llm_wrapper.generate(prompt)
    
#     # 5) extract and store memory
#     memory_text = memory_system.extract_memory(player_input, reply, llm_wrapper)
#     if memory_text:
#         memory_system.add_memory(memory_text, player_input, reply)
    
#     return reply, contexts, relevant_memories, prompt


# # -------------------- CLI --------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode", choices=["ingest", "chat"], help="ingest or chat")
#     parser.add_argument("path", nargs="?", help="path to data folder for ingest")
#     args = parser.parse_args()

#     if args.mode == "ingest":
#         if not args.path:
#             print("Provide a folder path to ingest, e.g. python rag_faiss_local.py ingest ./data")
#             sys.exit(1)
#         ingest_folder(args.path)
#         print("Ingest finished.")
#         sys.exit(0)

#     # chat mode
#     # initialize LLM wrapper
#     llm = None
#     if LLAMA_AVAILABLE and os.path.exists(LLAMA_MODEL_PATH):
#         try:
#             llm = LlamaWrapper(LLAMA_MODEL_PATH)
#             print("Using llama-cpp backend.")
#         except Exception as e:
#             print("Failed to init llama-cpp:", e)
#             llm = None

#     if llm is None and TRANSFORMERS_AVAILABLE:
#         try:
#             llm = HFWrapper(HF_MODEL_ID)
#             print("Using Transformers backend (fallback).")
#         except Exception as e:
#             print("Failed to init Transformers model:", e)
#             llm = None

#     if llm is None:
#         print("No LLM backend available. Install llama-cpp-python and/or transformers.")
#         sys.exit(1)

#     # Initialize memory system
#     memory_system = ConversationMemory()
#     print(f"Loaded {len(memory_system.memories)} existing memories.")

#     # basic persona and memory for testing
#     character_name = "Eri"
#     persona_notes = (
#         "Soft-spoken, quick-witted, composed but with a dry humor. "
#         "Speaks concisely, avoids giving out private secrets, and deflects questions she doesn't know. "
#         "You have an excellent memory and remember details about people you talk to."
#     )
#     short_memory = "She likes jasmine tea and keeps a sketchbook in the café."

#     print("Interactive chat with memory system. Type 'exit' to quit, 'memories' to view all memories.")
#     while True:
#         q = input("Player: ").strip()
#         if not q:
#             continue
#         if q.lower() in ("quit", "exit"):
#             break
#         if q.lower() == "memories":
#             print("\n=== All Memories ===")
#             for i, mem in enumerate(memory_system.memories, 1):
#                 print(f"{i}. {mem['memory_text']} (saved: {mem['timestamp'][:10]})")
#             print("===================\n")
#             continue
            
#         reply, contexts, memories, prompt = respond(q, character_name, persona_notes, 
#                                                    short_memory, llm, memory_system)
#         print(f"{character_name}: {reply}\n")
        
#         # Show which memories were used (optional, for debugging)
#         if memories:
#             print(f"[Used {len(memories)} relevant memories]")

# if __name__ == "__main__":
#     main()

# """
# rag_faiss_local.py

# - Ingest local text sources into a FAISS IndexIDMap with sentence-transformers embeddings.
# - Query the index for top-K contexts for RAG.
# - Generate replies via llama-cpp-python (preferred) or Transformers fallback (if llama-cpp not available).
# - Metadata mapping stored in JSON; index saved to disk.
# - **NEW**: Memory system that extracts and stores key moments from conversations

# Usage:
#   python rag_faiss_local.py ingest ./data   # create/update index from files in ./data
#   python rag_faiss_local.py chat            # interactive chat loop (uses built index)
# """

# import os
# import sys
# import json
# import time
# import math
# import glob
# import uuid
# import argparse
# from typing import List, Dict, Any
# from datetime import datetime
# from tqdm import tqdm

# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Try to import llama-cpp; if unavailable, we'll fallback to transformers
# LLAMA_AVAILABLE = False
# TRANSFORMERS_AVAILABLE = False
# try:
#     from llama_cpp import Llama
#     LLAMA_AVAILABLE = True
# except Exception:
#     LLAMA_AVAILABLE = False

# try:
#     from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#     TRANSFORMERS_AVAILABLE = True
# except Exception:
#     TRANSFORMERS_AVAILABLE = False

# # FAISS import
# try:
#     import faiss
# except Exception as e:
#     raise RuntimeError("faiss is required. Install faiss-cpu (pip) or faiss via conda on Windows.") from e

# # -------------------- CONFIG --------------------
# EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# EMBED_DIM = 384  # matches the model above
# FAISS_INDEX_PATH = "faiss_index.ivf"   # file to save FAISS index
# META_PATH = "meta_store.json"          # maps int_id -> metadata dict
# MEMORY_PATH = "conversation_memory.json"  # stores key memories from conversations

# # GPU Configuration
# USE_GPU = True  # Set to False to use CPU
# GPU_DEVICE = 0  # GPU device ID (0 for first GPU, 1 for second, etc.)

# # LLM config (edit)
# LLAMA_MODEL_PATH = "\\\\tholospg.itserv.scss.tcd.ie\\Pgrad\\mkhan5\\My Documents\\GitHub\\Visual-Novel-With-Custom-Character-Interactions-Using-LLM-RAG\\models\\mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # set if you have a ggml model for llama-cpp
# LLAMA_CTX = 2048
# LLAMA_TEMPERATURE = 0.7
# LLAMA_MAX_TOKENS = 256
# LLAMA_N_GPU_LAYERS = 35  # Number of layers to offload to GPU (0 = CPU only, -1 = all layers)

# # Transformers fallback (if no llama-cpp)
# HF_MODEL_ID = "gpt2"  # replace with a local bigger model if available (gpt2 is small, demo-only)
# HF_MAX_NEW_TOKENS = 256
# HF_TEMPERATURE = 0.7

# # RAG params
# TOP_K = 6
# CONTEXT_TOKEN_BUDGET = 1500  # approx tokens reserved for contexts (trim if exceed model ctx)
# MEMORY_TOP_K = 3  # number of relevant memories to include in prompts
# # ------------------------------------------------

# # Load embedding model once
# print("Loading embedding model:", EMBED_MODEL_NAME)
# device = "cuda" if USE_GPU else "cpu"
# print(f"Using device: {device}")
# embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)


# # -------------------- Utilities --------------------
# def save_json(path: str, obj: Any):
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, ensure_ascii=False, indent=2)


# def load_json(path: str) -> Any:
#     if not os.path.exists(path):
#         return {}
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# # Simple chunker: split text into chunks by paragraphs, then sliding-window merge to target size
# def chunk_text(text: str, max_chunk_chars: int = 800) -> List[str]:
#     paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
#     chunks: List[str] = []
#     current = ""
#     for p in paragraphs:
#         if len(current) + len(p) + 1 <= max_chunk_chars:
#             current = (current + "\n\n" + p).strip() if current else p
#         else:
#             if current:
#                 chunks.append(current)
#             if len(p) <= max_chunk_chars:
#                 current = p
#             else:
#                 # fallback: break by sentences if paragraph too long
#                 sentences = p.split(". ")
#                 cur2 = ""
#                 for s in sentences:
#                     if len(cur2) + len(s) + 2 <= max_chunk_chars:
#                         cur2 = (cur2 + ". " + s).strip() if cur2 else s
#                     else:
#                         if cur2:
#                             chunks.append(cur2.strip() + ".")
#                         cur2 = s
#                 if cur2:
#                     current = cur2.strip()
#                 else:
#                     current = ""
#     if current:
#         chunks.append(current)
#     return chunks


# # -------------------- FAISS INDEX helpers --------------------
# def create_faiss_index(emb_dim: int):
#     # Using IndexFlatIP + IndexIDMap for cosine (use normalized vectors)
#     index = faiss.IndexFlatIP(emb_dim)
    
#     # Move to GPU if available and enabled
#     if USE_GPU and faiss.get_num_gpus() > 0:
#         print(f"Moving FAISS index to GPU {GPU_DEVICE}")
#         res = faiss.StandardGpuResources()
#         index = faiss.index_cpu_to_gpu(res, GPU_DEVICE, index)
    
#     id_index = faiss.IndexIDMap(index)
#     return id_index


# def save_faiss(index, path: str):
#     faiss.write_index(index, path)
#     print("Saved faiss index to", path)


# def load_faiss(path: str):
#     if not os.path.exists(path):
#         return create_faiss_index(EMBED_DIM)
#     idx = faiss.read_index(path)
    
#     # Move to GPU if available and enabled
#     if USE_GPU and faiss.get_num_gpus() > 0:
#         print(f"Moving loaded FAISS index to GPU {GPU_DEVICE}")
#         res = faiss.StandardGpuResources()
#         # Extract base index if it's wrapped in IDMap
#         try:
#             base_idx = faiss.downcast_index(idx.index)
#             base_idx = faiss.index_cpu_to_gpu(res, GPU_DEVICE, base_idx)
#             idx = faiss.IndexIDMap(base_idx)
#         except Exception:
#             # If not IDMap, wrap it
#             idx = faiss.index_cpu_to_gpu(res, GPU_DEVICE, idx)
#             idx = faiss.IndexIDMap(idx)
#         return idx
    
#     # ensure it's wrapped in IDMap (it should be if you saved it that way)
#     try:
#         _ = idx.id_map
#         return idx
#     except Exception:
#         # wrap
#         return faiss.IndexIDMap(idx)


# # -------------------- Memory System --------------------
# class ConversationMemory:
#     """Stores and retrieves key moments from conversations"""
    
#     def __init__(self, memory_path: str = MEMORY_PATH):
#         self.memory_path = memory_path
#         self.memories = self._load_memories()
#         self.memory_embeddings = None
#         self.memory_index = None
#         self._build_memory_index()
    
#     def _load_memories(self) -> List[Dict]:
#         """Load memories from disk"""
#         if os.path.exists(self.memory_path):
#             with open(self.memory_path, "r", encoding="utf-8") as f:
#                 return json.load(f)
#         return []
    
#     def _save_memories(self):
#         """Save memories to disk"""
#         save_json(self.memory_path, self.memories)
    
#     def _build_memory_index(self):
#         """Build FAISS index for semantic search over memories"""
#         if not self.memories:
#             return
        
#         texts = [m["memory_text"] for m in self.memories]
#         embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
#         embeddings = embeddings.astype('float32')
        
#         # Normalize for cosine similarity
#         norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#         norms[norms == 0] = 1e-9
#         embeddings = embeddings / norms
        
#         self.memory_embeddings = embeddings
#         self.memory_index = faiss.IndexFlatIP(EMBED_DIM)
        
#         # Move to GPU if available
#         if USE_GPU and faiss.get_num_gpus() > 0:
#             res = faiss.StandardGpuResources()
#             self.memory_index = faiss.index_cpu_to_gpu(res, GPU_DEVICE, self.memory_index)
        
#         self.memory_index.add(embeddings)
    
#     def extract_memory(self, player_input: str, character_response: str, llm_wrapper) -> str:
#         """Use LLM to extract key information worth remembering"""
#         extraction_prompt = f"""Analyze this conversation exchange and extract any key facts, preferences, or important moments worth remembering. Focus on:
# - Personal preferences (favorite colors, foods, hobbies)
# - Important facts about the player
# - Significant events or revelations
# - Emotional moments or decisions

# If there's nothing particularly memorable, respond with "NONE".
# If there is something memorable, write a single concise sentence (under 20 words) summarizing what to remember.

# Player: {player_input}
# Character: {character_response}

# Memory to extract (or NONE):"""
        
#         try:
#             if isinstance(llm_wrapper, LlamaWrapper):
#                 memory = llm_wrapper.llm.create(
#                     prompt=extraction_prompt,
#                     max_tokens=50,
#                     temperature=0.3,
#                     stop=["\n"]
#                 )
#                 if isinstance(memory, dict):
#                     memory_text = memory.get("choices", [{}])[0].get("text", "").strip()
#                 else:
#                     memory_text = str(memory).strip()
#             else:
#                 memory_text = llm_wrapper.generate(extraction_prompt, max_tokens=50, temperature=0.3)
            
#             memory_text = memory_text.replace("Memory to extract:", "").strip()
            
#             if memory_text and memory_text.upper() != "NONE" and len(memory_text) > 5:
#                 return memory_text
#         except Exception as e:
#             print(f"Memory extraction error: {e}")
        
#         return None
    
#     def add_memory(self, memory_text: str, player_input: str, character_response: str):
#         """Add a new memory to the system"""
#         memory = {
#             "id": str(uuid.uuid4()),
#             "memory_text": memory_text,
#             "timestamp": datetime.now().isoformat(),
#             "context": {
#                 "player_input": player_input,
#                 "character_response": character_response
#             }
#         }
#         self.memories.append(memory)
#         self._save_memories()
#         self._build_memory_index()  # Rebuild index with new memory
#         print(f"[Memory saved: {memory_text}]")
    
#     def retrieve_relevant_memories(self, query: str, top_k: int = MEMORY_TOP_K) -> List[Dict]:
#         """Retrieve most relevant memories for current context"""
#         if not self.memories or self.memory_index is None:
#             return []
        
#         # Encode query
#         q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype('float32')
#         q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        
#         # Search
#         k = min(top_k, len(self.memories))
#         D, I = self.memory_index.search(np.expand_dims(q_emb, axis=0), k)
        
#         results = []
#         for score, idx in zip(D[0], I[0]):
#             if idx >= 0 and idx < len(self.memories):
#                 memory = self.memories[idx].copy()
#                 memory["relevance_score"] = float(score)
#                 results.append(memory)
        
#         return results
    
#     def get_all_memories_summary(self) -> str:
#         """Get a summary of all memories for context"""
#         if not self.memories:
#             return "No memories yet."
        
#         recent = self.memories[-5:]  # Last 5 memories
#         summary = "\n".join([f"- {m['memory_text']}" for m in recent])
#         return summary


# # -------------------- Ingest pipeline --------------------
# def ingest_folder(data_folder: str, index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH,
#                   batch_size: int = 64):
#     """
#     Walks data_folder for .txt, .jsonl, .json, .csv files and ingests textual items into FAISS.
#     - txt: entire file becomes one document
#     - jsonl/json: expects objects with 'id' and 'text' (or will generate an id)
#     - csv: first column is treated as text if header unknown
#     """
#     # load existing meta and index
#     meta = load_json(meta_path)  # maps str(int_id) -> metadata
#     index = load_faiss(index_path)

#     # build a set of used_int_ids to avoid collisions
#     used_int_ids = set(int(k) for k in meta.keys()) if meta else set()
#     next_int_id = max(used_int_ids) + 1 if used_int_ids else 1

#     # collect new documents as tuples (int_id, metadata, text)
#     new_docs = []

#     # helper to add doc
#     def add_doc(text, source_name, orig_id=None, metadata_extra=None):
#         nonlocal next_int_id
#         if not text or not text.strip():
#             return
#         chunks = chunk_text(text)
#         for i, c in enumerate(chunks):
#             int_id = next_int_id
#             next_int_id += 1
#             doc_id = orig_id or str(uuid.uuid4())
#             metadata = {
#                 "doc_id": doc_id,
#                 "source": source_name,
#                 "chunk_index": i,
#                 "text": c,
#             }
#             if metadata_extra:
#                 metadata.update(metadata_extra)
#             new_docs.append((int_id, metadata, c))

#     # walk files
#     patterns = ["*.txt", "*.jsonl", "*.json", "*.csv"]
#     file_list = []
#     for p in patterns:
#         file_list.extend(glob.glob(os.path.join(data_folder, p)))
#     print(f"Found {len(file_list)} files to ingest in {data_folder}")

#     for filepath in file_list:
#         ext = os.path.splitext(filepath)[1].lower()
#         name = os.path.basename(filepath)
#         try:
#             if ext == ".txt":
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     text = f.read()
#                 add_doc(text, source_name=name)
#             elif ext == ".jsonl":
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     for line in f:
#                         obj = json.loads(line)
#                         txt = obj.get("text") or obj.get("dialogue") or obj.get("content")
#                         add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
#             elif ext == ".json":
#                 with open(filepath, "r", encoding="utf-8") as f:
#                     data = json.load(f)
#                 # if it's a list of objects
#                 if isinstance(data, list):
#                     for obj in data:
#                         txt = obj.get("text") or obj.get("content")
#                         add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
#                 elif isinstance(data, dict):
#                     # try to find a key that is a list of items
#                     if "items" in data and isinstance(data["items"], list):
#                         for obj in data["items"]:
#                             txt = obj.get("text") or obj.get("content")
#                             add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
#                     else:
#                         # treat file as single doc
#                         txt = data.get("text") or json.dumps(data)
#                         add_doc(txt, source_name=name, orig_id=data.get("id"), metadata_extra=data)
#             elif ext == ".csv":
#                 import csv
#                 with open(filepath, newline='', encoding='utf-8') as csvfile:
#                     reader = csv.reader(csvfile)
#                     for row in reader:
#                         if not row:
#                             continue
#                         txt = row[0]
#                         add_doc(txt, source_name=name)
#             else:
#                 # ignore unknown
#                 pass
#         except Exception as e:
#             print("Failed to ingest", filepath, e)

#     if not new_docs:
#         print("No new docs found to ingest.")
#         return

#     # compute embeddings in batches
#     texts = [d[2] for d in new_docs]
#     int_ids = [d[0] for d in new_docs]
#     metas = {str(d[0]): d[1] for d in new_docs}

#     # batch encode
#     embeddings = []
#     for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
#         batch = texts[i:i + batch_size]
#         emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
#         embeddings.append(emb)
#     embeddings = np.vstack(embeddings).astype('float32')

#     # normalize for cosine similarity with inner-product index
#     norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
#     norms[norms == 0] = 1e-9
#     embeddings = embeddings / norms

#     # add to faiss index using add_with_ids
#     id_array = np.array(int_ids, dtype=np.int64)
#     index.add_with_ids(embeddings, id_array)
#     print(f"Added {len(int_ids)} vectors to FAISS index.")

#     # merge metadata and save
#     meta.update(metas)
#     save_json(meta_path, meta)
#     save_faiss(index, index_path)
#     print("Ingest complete.")


# # -------------------- Retrieval & prompt building --------------------
# def retrieve(query: str, top_k: int = TOP_K, index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH):
#     index = load_faiss(index_path)
#     meta = load_json(meta_path)
#     # encode
#     q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype('float32')
#     q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
#     D, I = index.search(np.expand_dims(q_emb, axis=0), top_k)
#     results = []
#     for dist, iid in zip(D[0], I[0]):
#         if int(iid) == -1:
#             continue
#         key = str(int(iid))
#         if key in meta:
#             entry = meta[key].copy()
#             entry["_score"] = float(dist)
#             results.append(entry)
#     return results


# def build_prompt(character_name: str, persona_notes: str, contexts: List[Dict], 
#                 memories: List[Dict], short_memory: str, player_input: str):
#     # contexts are ordered most relevant first
#     ctx_texts = []
#     for c in contexts:
#         s = f"[{c.get('source','unknown')}|chunk{c.get('chunk_index')}] {c.get('text')}"
#         ctx_texts.append(s)
#     contexts_joined = "\n\n---\n\n".join(ctx_texts)
    
#     # Format memories
#     memory_section = ""
#     if memories:
#         memory_texts = [f"- {m['memory_text']}" for m in memories]
#         memory_section = f"\n\nKey Memories (things you remember about the player):\n" + "\n".join(memory_texts)

#     prompt = f"""System: You are {character_name}. The following describes your voice, personality, and rules. Respond in-character, concisely, and do not invent facts beyond the Contexts or your Memories. If the player asks about something you don't know, politely deflect.

# Persona:
# {persona_notes}

# Contexts (most relevant first):
# {contexts_joined}{memory_section}

# Short memory:
# {short_memory or "None"}

# Player: {player_input}
# {character_name}:"""
#     return prompt


# # -------------------- LLM wrappers --------------------
# class LlamaWrapper:
#     def __init__(self, model_path: str, n_ctx: int = LLAMA_CTX):
#         if not LLAMA_AVAILABLE:
#             raise RuntimeError("llama-cpp-python not available")
#         print("Loading llama-cpp model:", model_path)
#         print(f"GPU layers to offload: {LLAMA_N_GPU_LAYERS}")
#         self.llm = Llama(
#             model_path=model_path, 
#             n_ctx=n_ctx,
#             n_gpu_layers=LLAMA_N_GPU_LAYERS  # Offload layers to GPU
#         )

#     def generate(self, prompt: str, max_tokens: int = LLAMA_MAX_TOKENS, temperature: float = LLAMA_TEMPERATURE):
#         # Use stop tokens to prevent it from continuing the dialogue
#         resp = self.llm.create_chat_completion(
#             prompt=prompt,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             stop=["Player:", "\nPlayer:", "\nSystem:"]
#         )
#         if isinstance(resp, dict):
#             choices = resp.get("choices")
#             if choices and isinstance(choices, list):
#                 txt = choices[0].get("text") or choices[0].get("message", {}).get("content", "")
#                 # Trim anything after it starts echoing the player
#                 txt = txt.split("Player:")[0].strip()
#                 return txt
#         return str(resp).split("Player:")[0].strip()

# class HFWrapper:
#     def __init__(self, model_id: str = HF_MODEL_ID):
#         if not TRANSFORMERS_AVAILABLE:
#             raise RuntimeError("Transformers not available")
#         print("Loading HF model:", model_id)
        
#         # Determine device
#         import torch
#         device = "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
#         print(f"HF model using device: {device}")
        
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_id, 
#             device_map="auto" if USE_GPU else None,
#             torch_dtype="auto" if USE_GPU else None
#         )
        
#         if not USE_GPU:
#             self.model = self.model.to(device)
        
#         self.pipe = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             device=0 if USE_GPU and device == "cuda" else -1
#         )

#     def generate(self, prompt: str, max_tokens: int = HF_MAX_NEW_TOKENS, temperature: float = HF_TEMPERATURE):
#         stop_seq = ["Player:", "\nPlayer:", "\nSystem:"]
#         out = self.pipe(
#             prompt,
#             max_new_tokens=max_tokens,
#             do_sample=True,
#             temperature=temperature,
#             return_full_text=False
#         )[0]["generated_text"]
#         # Stop if it starts to include the Player's turn
#         for stop in stop_seq:
#             if stop in out:
#                 out = out.split(stop)[0]
#                 break
#         return out.strip()


# # -------------------- Respond function --------------------
# def respond(player_input: str, character_name: str, persona_notes: str, short_memory: str,
#             llm_wrapper, memory_system: ConversationMemory, 
#             index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH,
#             top_k: int = TOP_K):
#     # 1) retrieve contexts
#     contexts = retrieve(player_input, top_k, index_path, meta_path)
    
#     # 2) retrieve relevant memories
#     relevant_memories = memory_system.retrieve_relevant_memories(player_input)
    
#     # 3) build prompt
#     prompt = build_prompt(character_name, persona_notes, contexts, relevant_memories, 
#                          short_memory, player_input)

#     # Basic prompt token budget enforcement (very rough: char -> token ~ 4 chars per token)
#     approx_tokens = len(prompt) / 4
#     if approx_tokens + LLAMA_MAX_TOKENS > LLAMA_CTX:
#         # trimming contexts: keep first N contexts until under budget
#         trimmed = []
#         tokens_used = len(f"System: You are {character_name}.") / 4
#         for c in contexts:
#             s = c.get("text", "")
#             t = len(s) / 4
#             if tokens_used + t + LLAMA_MAX_TOKENS < LLAMA_CTX:
#                 trimmed.append(c)
#                 tokens_used += t
#             else:
#                 break
#         contexts = trimmed
#         prompt = build_prompt(character_name, persona_notes, contexts, relevant_memories,
#                             short_memory, player_input)

#     # 4) generate response
#     reply = llm_wrapper.generate(prompt)
    
#     # 5) extract and store memory
#     memory_text = memory_system.extract_memory(player_input, reply, llm_wrapper)
#     if memory_text:
#         memory_system.add_memory(memory_text, player_input, reply)
    
#     return reply, contexts, relevant_memories, prompt


# # -------------------- CLI --------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("mode", choices=["ingest", "chat"], help="ingest or chat")
#     parser.add_argument("path", nargs="?", help="path to data folder for ingest")
#     args = parser.parse_args()

#     if args.mode == "ingest":
#         if not args.path:
#             print("Provide a folder path to ingest, e.g. python rag_faiss_local.py ingest ./data")
#             sys.exit(1)
#         ingest_folder(args.path)
#         print("Ingest finished.")
#         sys.exit(0)

#     # chat mode
#     # initialize LLM wrapper
#     llm = None
#     if LLAMA_AVAILABLE and os.path.exists(LLAMA_MODEL_PATH):
#         try:
#             llm = LlamaWrapper(LLAMA_MODEL_PATH)
#             print("Using llama-cpp backend.")
#         except Exception as e:
#             print("Failed to init llama-cpp:", e)
#             llm = None

#     if llm is None and TRANSFORMERS_AVAILABLE:
#         try:
#             llm = HFWrapper(HF_MODEL_ID)
#             print("Using Transformers backend (fallback).")
#         except Exception as e:
#             print("Failed to init Transformers model:", e)
#             llm = None

#     if llm is None:
#         print("No LLM backend available. Install llama-cpp-python and/or transformers.")
#         sys.exit(1)

#     # Initialize memory system
#     memory_system = ConversationMemory()
#     print(f"Loaded {len(memory_system.memories)} existing memories.")

#     # basic persona and memory for testing
#     character_name = "Eri"
#     persona_notes = (
#         "Soft-spoken, quick-witted, composed but with a dry humor. "
#         "Speaks concisely, avoids giving out private secrets, and deflects questions she doesn't know. "
#         "You have an excellent memory and remember details about people you talk to."
#     )
#     short_memory = "She likes jasmine tea and keeps a sketchbook in the café."

#     print("Interactive chat with memory system. Type 'exit' to quit, 'memories' to view all memories.")
#     while True:
#         q = input("Player: ").strip()
#         if not q:
#             continue
#         if q.lower() in ("quit", "exit"):
#             break
#         if q.lower() == "memories":
#             print("\n=== All Memories ===")
#             for i, mem in enumerate(memory_system.memories, 1):
#                 print(f"{i}. {mem['memory_text']} (saved: {mem['timestamp'][:10]})")
#             print("===================\n")
#             continue
            
#         reply, contexts, memories, prompt = respond(q, character_name, persona_notes, 
#                                                    short_memory, llm, memory_system)
#         print(f"{character_name}: {reply}\n")
        
#         # Show which memories were used (optional, for debugging)
#         if memories:
#             print(f"[Used {len(memories)} relevant memories]")

# if __name__ == "__main__":
#     main()

"""
rag_faiss_local.py

- Ingest local text sources into a FAISS IndexIDMap with sentence-transformers embeddings.
- Query the index for top-K contexts for RAG.
- Generate replies via llama-cpp-python (preferred) or Transformers fallback (if llama-cpp not available).
- Metadata mapping stored in JSON; index saved to disk.
- **NEW**: Memory system that extracts and stores key moments from conversations

Usage:
  python rag_faiss_local.py ingest ./data   # create/update index from files in ./data
  python rag_faiss_local.py chat            # interactive chat loop (uses built index)
"""

import os
import sys
import json
import time
import math
import glob
import uuid
import argparse
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm

import numpy as np
from sentence_transformers import SentenceTransformer

# Try to import llama-cpp; if unavailable, we'll fallback to transformers
LLAMA_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except Exception:
    LLAMA_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# FAISS import
try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss is required. Install faiss-cpu (pip) or faiss via conda on Windows.") from e

# -------------------- CONFIG --------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # matches the model above
FAISS_INDEX_PATH = "faiss_index.ivf"   # file to save FAISS index
META_PATH = "meta_store.json"          # maps int_id -> metadata dict
MEMORY_PATH = "conversation_memory.json"  # stores key memories from conversations

# GPU Configuration
USE_GPU = True  # Set to False to use CPU
GPU_DEVICE = 0  # GPU device ID (0 for first GPU, 1 for second, etc.)

# LLM config (edit)
LLAMA_MODEL_PATH = "\\\\tholospg.itserv.scss.tcd.ie\\Pgrad\\mkhan5\\My Documents\\GitHub\\Visual-Novel-With-Custom-Character-Interactions-Using-LLM-RAG\\models\\mistral-7b-instruct-v0.2.Q4_K_M.gguf"  # set if you have a ggml model for llama-cpp
LLAMA_CTX = 2048
LLAMA_TEMPERATURE = 0.7
LLAMA_MAX_TOKENS = 256
LLAMA_N_GPU_LAYERS = 35  # Number of layers to offload to GPU (0 = CPU only, -1 = all layers)

# Transformers fallback (if no llama-cpp)
HF_MODEL_ID = "gpt2"  # replace with a local bigger model if available (gpt2 is small, demo-only)
HF_MAX_NEW_TOKENS = 256
HF_TEMPERATURE = 0.7

# RAG params
TOP_K = 6
CONTEXT_TOKEN_BUDGET = 1500  # approx tokens reserved for contexts (trim if exceed model ctx)
MEMORY_TOP_K = 3  # number of relevant memories to include in prompts
# ------------------------------------------------

# Check GPU availability
import torch
CUDA_AVAILABLE = torch.cuda.is_available()
if USE_GPU and not CUDA_AVAILABLE:
    print("WARNING: GPU requested but CUDA not available. Falling back to CPU.")
    print("To enable GPU, install PyTorch with CUDA support:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    USE_GPU = False

# Load embedding model once
print("Loading embedding model:", EMBED_MODEL_NAME)
device = "cuda" if (USE_GPU and CUDA_AVAILABLE) else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)


# -------------------- Utilities --------------------
def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Simple chunker: split text into chunks by paragraphs, then sliding-window merge to target size
def chunk_text(text: str, max_chunk_chars: int = 800) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chunk_chars:
            current = (current + "\n\n" + p).strip() if current else p
        else:
            if current:
                chunks.append(current)
            if len(p) <= max_chunk_chars:
                current = p
            else:
                # fallback: break by sentences if paragraph too long
                sentences = p.split(". ")
                cur2 = ""
                for s in sentences:
                    if len(cur2) + len(s) + 2 <= max_chunk_chars:
                        cur2 = (cur2 + ". " + s).strip() if cur2 else s
                    else:
                        if cur2:
                            chunks.append(cur2.strip() + ".")
                        cur2 = s
                if cur2:
                    current = cur2.strip()
                else:
                    current = ""
    if current:
        chunks.append(current)
    return chunks


# -------------------- FAISS INDEX helpers --------------------
def create_faiss_index(emb_dim: int):
    # Using IndexFlatIP + IndexIDMap for cosine (use normalized vectors)
    index = faiss.IndexFlatIP(emb_dim)
    
    # Move to GPU if available and enabled
    if USE_GPU and CUDA_AVAILABLE and faiss.get_num_gpus() > 0:
        try:
            print(f"Moving FAISS index to GPU {GPU_DEVICE}")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, GPU_DEVICE, index)
        except Exception as e:
            print(f"WARNING: Failed to move FAISS to GPU: {e}")
            print("Continuing with CPU...")
    
    id_index = faiss.IndexIDMap(index)
    return id_index


def save_faiss(index, path: str):
    faiss.write_index(index, path)
    print("Saved faiss index to", path)


def load_faiss(path: str):
    if not os.path.exists(path):
        return create_faiss_index(EMBED_DIM)
    idx = faiss.read_index(path)
    
    # Move to GPU if available and enabled
    if USE_GPU and CUDA_AVAILABLE and faiss.get_num_gpus() > 0:
        try:
            print(f"Moving loaded FAISS index to GPU {GPU_DEVICE}")
            res = faiss.StandardGpuResources()
            # Extract base index if it's wrapped in IDMap
            try:
                base_idx = faiss.downcast_index(idx.index)
                base_idx = faiss.index_cpu_to_gpu(res, GPU_DEVICE, base_idx)
                idx = faiss.IndexIDMap(base_idx)
            except Exception:
                # If not IDMap, wrap it
                idx = faiss.index_cpu_to_gpu(res, GPU_DEVICE, idx)
                idx = faiss.IndexIDMap(idx)
            return idx
        except Exception as e:
            print(f"WARNING: Failed to move FAISS to GPU: {e}")
            print("Continuing with CPU...")
    
    # ensure it's wrapped in IDMap (it should be if you saved it that way)
    try:
        _ = idx.id_map
        return idx
    except Exception:
        # wrap
        return faiss.IndexIDMap(idx)


# -------------------- Memory System --------------------
class ConversationMemory:
    """Stores and retrieves key moments from conversations"""
    
    def __init__(self, memory_path: str = MEMORY_PATH):
        self.memory_path = memory_path
        self.memories = self._load_memories()
        self.memory_embeddings = None
        self.memory_index = None
        self._build_memory_index()
    
    def _load_memories(self) -> List[Dict]:
        """Load memories from disk"""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def _save_memories(self):
        """Save memories to disk"""
        save_json(self.memory_path, self.memories)
    
    def _build_memory_index(self):
        """Build FAISS index for semantic search over memories"""
        if not self.memories:
            return
        
        texts = [m["memory_text"] for m in self.memories]
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        embeddings = embeddings / norms
        
        self.memory_embeddings = embeddings
        self.memory_index = faiss.IndexFlatIP(EMBED_DIM)
        
        # Move to GPU if available
        if USE_GPU and CUDA_AVAILABLE and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.memory_index = faiss.index_cpu_to_gpu(res, GPU_DEVICE, self.memory_index)
            except Exception as e:
                print(f"WARNING: Failed to move memory index to GPU: {e}")
        
        self.memory_index.add(embeddings)
    
    def extract_memory(self, player_input: str, character_response: str, llm_wrapper) -> str:
        """Use LLM to extract key information worth remembering"""
        extraction_prompt = f"""Analyze this conversation exchange and extract any key facts, preferences, or important moments worth remembering. Focus on:
- Personal preferences (favorite colors, foods, hobbies)
- Important facts about the player
- Significant events or revelations
- Emotional moments or decisions

If there's nothing particularly memorable, respond with "NONE".
If there is something memorable, write a single concise sentence (under 20 words) summarizing what to remember.

Player: {player_input}
Character: {character_response}

Memory to extract (or NONE):"""
        
        try:
            if isinstance(llm_wrapper, LlamaWrapper):
                try:
                    # Try newer API first
                    memory = llm_wrapper.llm(
                        extraction_prompt,
                        max_tokens=50,
                        temperature=0.3,
                        stop=["\n"]
                    )
                except TypeError:
                    # Fallback to older API
                    memory = llm_wrapper.llm.create_completion(
                        extraction_prompt,
                        max_tokens=50,
                        temperature=0.3,
                        stop=["\n"]
                    )
                
                if isinstance(memory, dict):
                    memory_text = memory.get("choices", [{}])[0].get("text", "").strip()
                else:
                    memory_text = str(memory).strip()
            else:
                memory_text = llm_wrapper.generate(extraction_prompt, max_tokens=50, temperature=0.3)
            
            memory_text = memory_text.replace("Memory to extract:", "").strip()
            
            if memory_text and memory_text.upper() != "NONE" and len(memory_text) > 5:
                return memory_text
        except Exception as e:
            print(f"Memory extraction error: {e}")
        
        return None
    
    def add_memory(self, memory_text: str, player_input: str, character_response: str):
        """Add a new memory to the system"""
        memory = {
            "id": str(uuid.uuid4()),
            "memory_text": memory_text,
            "timestamp": datetime.now().isoformat(),
            "context": {
                "player_input": player_input,
                "character_response": character_response
            }
        }
        self.memories.append(memory)
        self._save_memories()
        self._build_memory_index()  # Rebuild index with new memory
        print(f"[Memory saved: {memory_text}]")
    
    def retrieve_relevant_memories(self, query: str, top_k: int = MEMORY_TOP_K) -> List[Dict]:
        """Retrieve most relevant memories for current context"""
        if not self.memories or self.memory_index is None:
            return []
        
        # Encode query
        q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype('float32')
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        
        # Search
        k = min(top_k, len(self.memories))
        D, I = self.memory_index.search(np.expand_dims(q_emb, axis=0), k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx >= 0 and idx < len(self.memories):
                memory = self.memories[idx].copy()
                memory["relevance_score"] = float(score)
                results.append(memory)
        
        return results
    
    def get_all_memories_summary(self) -> str:
        """Get a summary of all memories for context"""
        if not self.memories:
            return "No memories yet."
        
        recent = self.memories[-5:]  # Last 5 memories
        summary = "\n".join([f"- {m['memory_text']}" for m in recent])
        return summary


# -------------------- Ingest pipeline --------------------
def ingest_folder(data_folder: str, index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH,
                  batch_size: int = 64):
    """
    Walks data_folder for .txt, .jsonl, .json, .csv files and ingests textual items into FAISS.
    - txt: entire file becomes one document
    - jsonl/json: expects objects with 'id' and 'text' (or will generate an id)
    - csv: first column is treated as text if header unknown
    """
    # load existing meta and index
    meta = load_json(meta_path)  # maps str(int_id) -> metadata
    index = load_faiss(index_path)

    # build a set of used_int_ids to avoid collisions
    used_int_ids = set(int(k) for k in meta.keys()) if meta else set()
    next_int_id = max(used_int_ids) + 1 if used_int_ids else 1

    # collect new documents as tuples (int_id, metadata, text)
    new_docs = []

    # helper to add doc
    def add_doc(text, source_name, orig_id=None, metadata_extra=None):
        nonlocal next_int_id
        if not text or not text.strip():
            return
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            int_id = next_int_id
            next_int_id += 1
            doc_id = orig_id or str(uuid.uuid4())
            metadata = {
                "doc_id": doc_id,
                "source": source_name,
                "chunk_index": i,
                "text": c,
            }
            if metadata_extra:
                metadata.update(metadata_extra)
            new_docs.append((int_id, metadata, c))

    # walk files
    patterns = ["*.txt", "*.jsonl", "*.json", "*.csv"]
    file_list = []
    for p in patterns:
        file_list.extend(glob.glob(os.path.join(data_folder, p)))
    print(f"Found {len(file_list)} files to ingest in {data_folder}")

    for filepath in file_list:
        ext = os.path.splitext(filepath)[1].lower()
        name = os.path.basename(filepath)
        try:
            if ext == ".txt":
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                add_doc(text, source_name=name)
            elif ext == ".jsonl":
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        txt = obj.get("text") or obj.get("dialogue") or obj.get("content")
                        add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
            elif ext == ".json":
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # if it's a list of objects
                if isinstance(data, list):
                    for obj in data:
                        txt = obj.get("text") or obj.get("content")
                        add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
                elif isinstance(data, dict):
                    # try to find a key that is a list of items
                    if "items" in data and isinstance(data["items"], list):
                        for obj in data["items"]:
                            txt = obj.get("text") or obj.get("content")
                            add_doc(txt, source_name=name, orig_id=obj.get("id"), metadata_extra=obj)
                    else:
                        # treat file as single doc
                        txt = data.get("text") or json.dumps(data)
                        add_doc(txt, source_name=name, orig_id=data.get("id"), metadata_extra=data)
            elif ext == ".csv":
                import csv
                with open(filepath, newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if not row:
                            continue
                        txt = row[0]
                        add_doc(txt, source_name=name)
            else:
                # ignore unknown
                pass
        except Exception as e:
            print("Failed to ingest", filepath, e)

    if not new_docs:
        print("No new docs found to ingest.")
        return

    # compute embeddings in batches
    texts = [d[2] for d in new_docs]
    int_ids = [d[0] for d in new_docs]
    metas = {str(d[0]): d[1] for d in new_docs}

    # batch encode
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i + batch_size]
        emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype('float32')

    # normalize for cosine similarity with inner-product index
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms

    # add to faiss index using add_with_ids
    id_array = np.array(int_ids, dtype=np.int64)
    index.add_with_ids(embeddings, id_array)
    print(f"Added {len(int_ids)} vectors to FAISS index.")

    # merge metadata and save
    meta.update(metas)
    save_json(meta_path, meta)
    save_faiss(index, index_path)
    print("Ingest complete.")


# -------------------- Retrieval & prompt building --------------------
def retrieve(query: str, top_k: int = TOP_K, index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH):
    index = load_faiss(index_path)
    meta = load_json(meta_path)
    # encode
    q_emb = embedder.encode([query], convert_to_numpy=True)[0].astype('float32')
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    D, I = index.search(np.expand_dims(q_emb, axis=0), top_k)
    results = []
    for dist, iid in zip(D[0], I[0]):
        if int(iid) == -1:
            continue
        key = str(int(iid))
        if key in meta:
            entry = meta[key].copy()
            entry["_score"] = float(dist)
            results.append(entry)
    return results


def build_prompt(character_name: str, persona_notes: str, contexts: List[Dict], 
                memories: List[Dict], short_memory: str, player_input: str):
    # contexts are ordered most relevant first
    ctx_texts = []
    for c in contexts:
        s = f"[{c.get('source','unknown')}|chunk{c.get('chunk_index')}] {c.get('text')}"
        ctx_texts.append(s)
    contexts_joined = "\n\n---\n\n".join(ctx_texts)
    
    # Format memories
    memory_section = ""
    if memories:
        memory_texts = [f"- {m['memory_text']}" for m in memories]
        memory_section = f"\n\nKey Memories (things you remember about the player):\n" + "\n".join(memory_texts)

    prompt = f"""System: You are {character_name}. The following describes your voice, personality, and rules. Respond in-character, concisely, and do not invent facts beyond the Contexts or your Memories. If the player asks about something you don't know, politely deflect.

Persona:
{persona_notes}

Contexts (most relevant first):
{contexts_joined}{memory_section}

Short memory:
{short_memory or "None"}

Player: {player_input}
{character_name}:"""
    return prompt


# -------------------- LLM wrappers --------------------
class LlamaWrapper:
    def __init__(self, model_path: str, n_ctx: int = LLAMA_CTX):
        if not LLAMA_AVAILABLE:
            raise RuntimeError("llama-cpp-python not available")
        print("Loading llama-cpp model:", model_path)
        print(f"GPU layers to offload: {LLAMA_N_GPU_LAYERS}")
        self.llm = Llama(
            model_path=model_path, 
            n_ctx=n_ctx,
            n_gpu_layers=LLAMA_N_GPU_LAYERS  # Offload layers to GPU
        )

    def generate(self, prompt: str, max_tokens: int = LLAMA_MAX_TOKENS, temperature: float = LLAMA_TEMPERATURE):
        # Use stop tokens to prevent it from continuing the dialogue
        try:
            # Try the newer API first (llama-cpp-python >= 0.2.0)
            resp = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["Player:", "\nPlayer:", "\nSystem:"]
            )
        except TypeError:
            # Fallback to older API
            resp = self.llm.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["Player:", "\nPlayer:", "\nSystem:"]
            )
        
        if isinstance(resp, dict):
            choices = resp.get("choices")
            if choices and isinstance(choices, list):
                txt = choices[0].get("text") or choices[0].get("message", {}).get("content", "")
                # Trim anything after it starts echoing the player
                txt = txt.split("Player:")[0].strip()
                return txt
        return str(resp).split("Player:")[0].strip()

class HFWrapper:
    def __init__(self, model_id: str = HF_MODEL_ID):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available")
        print("Loading HF model:", model_id)
        
        # Determine device
        device = "cuda" if (USE_GPU and CUDA_AVAILABLE) else "cpu"
        print(f"HF model using device: {device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if USE_GPU and CUDA_AVAILABLE:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                device_map="auto",
                torch_dtype="auto"
            )
            device_id = 0
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
            self.model = self.model.to(device)
            device_id = -1
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_id
        )

    def generate(self, prompt: str, max_tokens: int = HF_MAX_NEW_TOKENS, temperature: float = HF_TEMPERATURE):
        stop_seq = ["Player:", "\nPlayer:", "\nSystem:"]
        out = self.pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            return_full_text=False
        )[0]["generated_text"]
        # Stop if it starts to include the Player's turn
        for stop in stop_seq:
            if stop in out:
                out = out.split(stop)[0]
                break
        return out.strip()


# -------------------- Respond function --------------------
def respond(player_input: str, character_name: str, persona_notes: str, short_memory: str,
            llm_wrapper, memory_system: ConversationMemory, 
            index_path: str = FAISS_INDEX_PATH, meta_path: str = META_PATH,
            top_k: int = TOP_K):
    # 1) retrieve contexts
    contexts = retrieve(player_input, top_k, index_path, meta_path)
    
    # 2) retrieve relevant memories
    relevant_memories = memory_system.retrieve_relevant_memories(player_input)
    
    # 3) build prompt
    prompt = build_prompt(character_name, persona_notes, contexts, relevant_memories, 
                         short_memory, player_input)

    # Basic prompt token budget enforcement (very rough: char -> token ~ 4 chars per token)
    approx_tokens = len(prompt) / 4
    if approx_tokens + LLAMA_MAX_TOKENS > LLAMA_CTX:
        # trimming contexts: keep first N contexts until under budget
        trimmed = []
        tokens_used = len(f"System: You are {character_name}.") / 4
        for c in contexts:
            s = c.get("text", "")
            t = len(s) / 4
            if tokens_used + t + LLAMA_MAX_TOKENS < LLAMA_CTX:
                trimmed.append(c)
                tokens_used += t
            else:
                break
        contexts = trimmed
        prompt = build_prompt(character_name, persona_notes, contexts, relevant_memories,
                            short_memory, player_input)

    # 4) generate response
    reply = llm_wrapper.generate(prompt)
    
    # 5) extract and store memory
    memory_text = memory_system.extract_memory(player_input, reply, llm_wrapper)
    if memory_text:
        memory_system.add_memory(memory_text, player_input, reply)
    
    return reply, contexts, relevant_memories, prompt


# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ingest", "chat"], help="ingest or chat")
    parser.add_argument("path", nargs="?", help="path to data folder for ingest")
    args = parser.parse_args()

    if args.mode == "ingest":
        if not args.path:
            print("Provide a folder path to ingest, e.g. python rag_faiss_local.py ingest ./data")
            sys.exit(1)
        ingest_folder(args.path)
        print("Ingest finished.")
        sys.exit(0)

    # chat mode
    # initialize LLM wrapper
    llm = None
    if LLAMA_AVAILABLE and os.path.exists(LLAMA_MODEL_PATH):
        try:
            llm = LlamaWrapper(LLAMA_MODEL_PATH)
            print("Using llama-cpp backend.")
        except Exception as e:
            print("Failed to init llama-cpp:", e)
            llm = None

    if llm is None and TRANSFORMERS_AVAILABLE:
        try:
            llm = HFWrapper(HF_MODEL_ID)
            print("Using Transformers backend (fallback).")
        except Exception as e:
            print("Failed to init Transformers model:", e)
            llm = None

    if llm is None:
        print("No LLM backend available. Install llama-cpp-python and/or transformers.")
        sys.exit(1)

    # Initialize memory system
    memory_system = ConversationMemory()
    print(f"Loaded {len(memory_system.memories)} existing memories.")

    # basic persona and memory for testing
    character_name = "Eri"
    persona_notes = (
        "Soft-spoken, quick-witted, composed but with a dry humor. "
        "Speaks concisely, avoids giving out private secrets, and deflects questions she doesn't know. "
        "You have an excellent memory and remember details about people you talk to."
    )
    short_memory = "She likes jasmine tea and keeps a sketchbook in the café."

    print("Interactive chat with memory system. Type 'exit' to quit, 'memories' to view all memories.")
    while True:
        q = input("Player: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "memories":
            print("\n=== All Memories ===")
            for i, mem in enumerate(memory_system.memories, 1):
                print(f"{i}. {mem['memory_text']} (saved: {mem['timestamp'][:10]})")
            print("===================\n")
            continue
            
        reply, contexts, memories, prompt = respond(q, character_name, persona_notes, 
                                                   short_memory, llm, memory_system)
        print(f"{character_name}: {reply}\n")
        
        # Show which memories were used (optional, for debugging)
        if memories:
            print(f"[Used {len(memories)} relevant memories]")

if __name__ == "__main__":
    main()
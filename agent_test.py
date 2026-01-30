from __future__ import annotations

from typing import List, Dict, Any, Optional, TypedDict, Literal
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
import os
import re
from pathlib import Path
from collections import defaultdict
import bisect


# 1) LLM
from dotenv import load_dotenv
load_dotenv()
# LLM (example)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Milvus vector store (choose one that matches your environment)
from langchain_milvus import Milvus, BM25BuiltInFunction

# LangGraph (recommended for agentic workflows)
from langgraph.graph import StateGraph, END

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# 1) Schemas (structured outputs)
# ---------------------------

class SearchPlan(BaseModel):
    """LLM proposes search areas and concrete queries."""
    areas: List[str] = Field(..., description="High-level areas to search (privacy, fairness, etc.)")
    queries: List[str] = Field(..., description="Concrete search queries for vector DB")


class RelevanceDecision(BaseModel):
    relevant: bool = Field(..., description="Whether this chunk is relevant to scenario")
    reason: str = Field(..., description="Short reason grounded in the chunk")
    key_points: List[str] = Field(default_factory=list, description="Key ethical points extracted")


class EthicalConsideration(BaseModel):
    id: str
    bullet: str
    supporting_chunks: list[str] = Field(default_factory=list)


class QAPair(BaseModel):
    question: str
    answer: str
    linked_consideration_ids: List[str] = Field(default_factory=list)


class QAList(BaseModel):
    qa_pairs: List[QAPair]


class ScenarioOut(BaseModel):
    """Normalized scenario output sent to the frontend."""
    name: str


class FinalOutput(BaseModel):
    scenario: ScenarioOut
    search_plan: SearchPlan
    considerations: List[EthicalConsideration]
    qa_pairs: List[QAPair]


# ---------------------------
# 2) State for LangGraph
# ---------------------------

class State(TypedDict):
    scenario: str
    search_plan: Optional[SearchPlan]
    retrieved_chunks: List[Document]              # raw retrieved
    candidate_chunks: List[Document]              # deduped / merged
    relevant_chunks: List[Dict[str, Any]]         # [{doc, decision}]
    considerations: List[EthicalConsideration]
    qa_pairs: List[QAPair]


# ---------------------------
# 3) Prompts
# ---------------------------

plan_parser = PydanticOutputParser(pydantic_object=SearchPlan)
plan_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert in applied AI ethics and compliance. "
     "Given a scenario, propose search areas and vector-search queries to retrieve relevant policy/guidance. "
     "Generate the concrete queries in Norwegian (norsk). "
     "Return ONLY valid JSON matching the schema.\n{format_instructions}"),
    ("human", "Scenario:\n{scenario}\n\nConstraints:\n- Provide 6-12 areas\n- Provide 10-25 diverse queries\n- Queries should be short (3-12 words) and cover different angles\n- Queries should be in Norwegian (norsk)")
]).partial(format_instructions=plan_parser.get_format_instructions())


rel_parser = PydanticOutputParser(pydantic_object=RelevanceDecision)
relevance_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance filter. Decide if the provided chunk contains information that is "
     "directly useful for ethical considerations in the given scenario. "
     "Be conservative: mark relevant=true only if it clearly applies.\n"
     "Note: The scenario will be provided in English and chunk text may be in Norwegian; evaluate relevance across languages.\n"
     "Return ONLY valid JSON matching the schema.\n{format_instructions}"),
    ("human",
     "Scenario:\n{scenario}\n\nChunk metadata:\n{metadata}\n\nChunk text:\n{chunk_text}")
]).partial(format_instructions=rel_parser.get_format_instructions())


# Summarize considerations from relevant chunks
considerations_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You synthesize ethical considerations grounded in the provided relevant excerpts. "
     "Write concise bullets in English. Avoid adding anything not supported by excerpts. "
     "Where possible, attach supporting chunk_ids."),
    ("human",
     "Scenario:\n{scenario}\n\nRelevant excerpts:\n{excerpts}\n\n"
     "Output format:\n"
     "- Return 8-20 bullets\n"
     "- Each bullet should be actionable and specific\n"
     "- Each bullet should mention the ethical dimension (privacy, fairness, transparency, safety, etc.)\n"
     "- After each bullet, include (support: chunk_id1, chunk_id2) if available.\n"
     "- Return bullets in English.")
])


# QA prompt: require structured JSON output matching QAList schema
qa_parser = PydanticOutputParser(pydantic_object=QAList)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Generate high-quality Ethical QA pairs from the considerations. "
     "Questions should test compliance and real-world decision making. "
     "Answers should be short but complete, including what to do / what to check / what to document. "
     "Do not invent new considerations.\n"
     "Output language: English. Final QA should be in English.\n"
     "Return ONLY valid JSON matching this schema: {format_instructions}"),
    ("human",
     "Scenario:\n{scenario}\n\nEthical considerations:\n{considerations}\n\n"
     "Generate 10-25 QA pairs in English. Include edge cases and 'what if' questions.")
]).partial(format_instructions=qa_parser.get_format_instructions())


# ---------------------------
# 4) Utilities
# ---------------------------
def doc_to_chunk_id(doc: Document) -> str:
    md = doc['metadata'] or {}
    source = md.get("source", "unknown_source")
    source = os.path.splitext(os.path.basename(source))[0]
    start = md.get("start_line", "?")
    end = md.get("end_line", "?")
    return f"{source}|L{start}-{end}"

def format_excerpts(relevant: List[Dict[str, Any]]) -> str:
    blocks = []
    for item in relevant:
        doc: Document = item["doc"]
        cid = doc_to_chunk_id(doc)
        md = doc['metadata'] or {}
        header = f"[chunk_id={cid}] doc={md.get('doc_name')} page={md.get('page')} "
        blocks.append(header + "\n" + doc['page_content'].strip())
    return "\n\n---\n\n".join(blocks)


def clean_page_text(text: str) -> str:
    """Remove long separator/table lines and collapse repeated punctuation.

    Heuristics:
    - Drop lines that are mostly non-alphanumeric (e.g., long runs of '-', '|', '_') and at least 30 chars long.
    - Collapse very long repeated dash/underscore runs inside lines.
    - Collapse multiple consecutive blank lines to one.
    """
    lines = text.splitlines()
    cleaned_lines: List[str] = []
    for ln in lines:
        stripped = ln.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        # If the line is long and mostly non-alphanumeric punctuation, drop it
        if len(stripped) >= 30:
            non_alnum = sum(1 for c in stripped if not (c.isalnum() or c.isspace()))
            ratio = non_alnum / len(stripped)
            if ratio > 0.6 or re.match(r'^[\|\-_=~`\s]+$', stripped):
                # This is very likely a table border/separator — skip it
                continue
        # Collapse runs of repeated dashes/underscores to a single character (keep lines readable)
        ln = re.sub(r'([-—])\1{4,}', r'\1', ln)
        ln = re.sub(r'_{4,}', '_', ln)
        cleaned_lines.append(ln)

    # Collapse multiple blank lines to a single blank line
    out_lines: List[str] = []
    prev_blank = False
    for ln in cleaned_lines:
        if ln.strip() == '':
            if not prev_blank:
                out_lines.append('')
            prev_blank = True
        else:
            out_lines.append(ln)
            prev_blank = False

    return "\n".join(out_lines).strip()


def dedupe_docs(docs: List[Document], idea, chunk_word_budget=200) -> List[Document]:
    deduped = sorted(
                {d.metadata["pk"]: (d, s*100) for d, s in docs}.values(),
                key=lambda x: x[1],
                reverse=False)
    deduped_filtered = [e[0] for e in deduped if e[1] > 10.0]
    print("dedupe_docs: removed number of duplicates and insignificant:", len(docs) - len(deduped_filtered))

    # Minimal logic: emit one compact word window per deduped document (no per-source merging).
    chunks: List[Dict[str, Any]] = []
    pre = int(chunk_word_budget * 0.2)
    post = chunk_word_budget - pre

    for d in deduped_filtered:
        md = d.metadata or {}
        src = md.get('source') or md.get('doc_name')

        # If source is missing, fall back to using the doc's inline content so we keep chunk counts equal
        if not src:
            logger.warning("Source not specified for doc with pk=%s; using inline content", md.get("pk"))
            page_text = d.get('page_content', '').strip()
            chunks.append({"metadata": md, "page_content": page_text})
            continue

        path = Path(src)
        if not path.exists():
            alt = Path('documents') / src
            if alt.exists():
                path = alt
            else:
                logger.warning("Source file not found: %s; using inline content if available", src)
                page_text = d.get('page_content', '').strip()
                chunks.append({"metadata": md, "page_content": page_text})
                continue

        lines = clean_page_text(path.read_text(encoding='utf-8')).splitlines()
        cum = [0]
        for ln in lines:
            cum.append(cum[-1] + len(re.findall(r"\S+", ln)))
        total = cum[-1] or 1

        s = int(md.get('start_line', 1) or 1)
        e = int(md.get('end_line', s) or s)
        s = max(1, min(s, len(lines))); e = max(1, min(e, len(lines)))
        sw = cum[s - 1] + 1; ew = cum[e]

        if (ew - sw + 1) >= chunk_word_budget:
            w0 = sw; w1 = min(total, w0 + chunk_word_budget - 1)
        else:
            w0 = max(1, sw - pre); w1 = min(total, ew + post)

        ws = max(1, bisect.bisect_left(cum, w0))
        we = min(len(lines), bisect.bisect_left(cum, w1))
        page_text = "\n".join(lines[ws - 1: we])
        chunks.append({"metadata": {"source": str(path), "start_line": ws, "end_line": we, "pk": md.get("pk")}, "page_content": page_text})

    # Ensure we return a chunk for each deduped document (fallbacks above preserve count)
    return chunks



# ---------------------------
# 5) Nodes
# ---------------------------

def make_plan(state: State, llm) -> State:
    scenario = state["scenario"]
    logger.info("Stage: plan — generating search plan for scenario: %s", scenario)
    resp = llm.invoke(plan_prompt.format_messages(scenario=scenario))
    plan = plan_parser.parse(resp.content)
    logger.info("Plan generated: %d areas, %d queries", len(plan.areas), len(plan.queries))
    return {**state, "search_plan": plan}

def retrieve(state: State, retriever, query_parallelism: int = 36) -> State:
    plan = state["search_plan"]
    assert plan is not None
    all_docs: List[Document] = []
    logger.info("Stage: retrieve — running %d queries", len(plan.queries))
    # Query expansion: do multiple searches, merge results
    print("plan.queries:\n", plan.queries)

    queries = plan.queries
    max_workers = min(max(1, query_parallelism), max(1, len(queries)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_query = {
            ex.submit(
                vectorstore.similarity_search_with_score,
                q,
                k=10,
                ranker_type="weighted",
                ranker_params={"weights": [0.65, 0.35]},
                param=[
                    {"metric_type": "L2", "params": {"nprobe": 16}},  # dense
                    {"metric_type": "BM25", "params": {}},            # sparse
                ],
            ): q
            for q in queries
        }

        for fut in as_completed(future_to_query):
            q = future_to_query[fut]
            try:
                similar_chunks = fut.result()
                count = len(similar_chunks) if hasattr(similar_chunks, "__len__") else 1
                logger.info("Query '%s': retrieved %d chunks", q, count)
                if isinstance(similar_chunks, list):
                    all_docs.extend(similar_chunks)
                else:
                    all_docs.append(similar_chunks)
            except Exception as e:
                logger.exception("Query '%s' failed: %s", q, e)

    all_docs = dedupe_docs(all_docs, None)
    logger.info("Retrieved %d unique chunks", len(all_docs))
    return {**state, "retrieved_chunks": all_docs, "candidate_chunks": all_docs}

def filter_relevance(state: State, llm, max_chunks: int = 200, llm_concurrency: int = 36) -> State:
    scenario = state["scenario"]
    candidates = state["candidate_chunks"][:max_chunks]  # control cost
    logger.info("Stage: filter_relevance — evaluating %d candidates (limiting to %d)", len(candidates), max_chunks)
    relevant: List[Dict[str, Any]] = []

    if not candidates:
        logger.info("No candidates to evaluate")
        return {**state, "relevant_chunks": relevant}

    max_workers = min(max(1, llm_concurrency), len(candidates))
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, doc in enumerate(candidates, 1):
            md = doc['metadata'] or {}
            logger.debug("Scheduling evaluation candidate %d/%d: chunk_id=%s", i, len(candidates), doc_to_chunk_id(doc))
            msg = relevance_prompt.format_messages(
                scenario=scenario,
                metadata=str(md),
                chunk_text=doc['page_content'][:2000]  # keep bounded
            )
            futures[ex.submit(llm.invoke, msg)] = doc

        for fut in as_completed(futures):
            doc = futures[fut]
            try:
                resp = fut.result()
                decision = rel_parser.parse(resp.content)
                logger.debug("Decision for %s: relevant=%s, reason=%s", doc_to_chunk_id(doc), decision.relevant, (decision.reason or '')[:150])
                if decision.relevant:
                    relevant.append({"doc": doc, "decision": decision})
            except Exception as e:
                logger.exception("Error evaluating candidate %s: %s", doc_to_chunk_id(doc), e)
    logger.info("Filter complete — %d relevant chunks found", len(relevant))
    return {**state, "relevant_chunks": relevant} 

def summarize_considerations(state: State, llm) -> State:
    scenario = state["scenario"]
    relevant = state["relevant_chunks"]
    logger.info("Stage: summarize_considerations — synthesizing from %d relevant excerpts", len(relevant))

    excerpts = format_excerpts(relevant)
    resp = llm.invoke(considerations_prompt.format_messages(
        scenario=scenario,
        excerpts=excerpts[:12000]  # bound token usage; for large sets do map-reduce
    ))

    # Parse bullets heuristically into EthicalConsideration objects
    lines = [ln.strip() for ln in resp.content.split("\n") if ln.strip().startswith("-")]
    logger.info("LLM returned %d bullet lines", len(lines))
    considerations: List[EthicalConsideration] = []
    for i, ln in enumerate(lines, start=1):
        # Parse trailing metadata like "(support: chunk_id1, chunk_id2)"
        bullet = ln[1:].strip()
        support_ids: List[str] = []
        # Match a trailing parenthetical metadata and allow for trailing punctuation after it (e.g. ").")
        m = re.search(r"\(([^)]*)\)\s*[.?!,;:]*\s*$", bullet)
        if m:
            meta = m.group(1)
            bullet = bullet[:m.start()].strip()
            # Remove any trailing punctuation leftover from the sentence end
            bullet = bullet.rstrip(' .,:;!?:')
            # Split into key/value parts using ';' and parse known keys
            for part in [p.strip() for p in meta.split(';') if p.strip()]:
                if ':' in part:
                    key, val = part.split(':', 1)
                    key = key.strip().lower()
                    val = val.strip()
                    if key == 'support':
                        support_ids = [s.strip().rstrip('.') for s in re.split('[,;]', val) if s.strip()]
        considerations.append(EthicalConsideration(
            id=f"C{i:02d}",
            bullet=bullet,
            supporting_chunks=support_ids,
        ))
    logger.info("Parsed %d considerations", len(considerations))
    return {**state, "considerations": considerations} 

def generate_qa(state: State, llm) -> State:
    scenario = state["scenario"]
    cons = state["considerations"]
    logger.info("Stage: generate_qa — creating QA pairs from %d considerations", len(cons))

    cons_text = "\n".join([f"{c.id}: {c.bullet}" for c in cons])
    resp = llm.invoke(qa_prompt.format_messages(
        scenario=scenario,
        considerations=cons_text
    ))

    # Parse structured JSON output using Pydantic
    parsed = qa_parser.parse(resp.content)
    qa_pairs: List[QAPair] = parsed.qa_pairs

    logger.info("Generated %d QA pairs", len(qa_pairs))
    return {**state, "qa_pairs": qa_pairs} 

def assemble(state: State) -> FinalOutput:
    logger.info("Stage: assemble — assembling final output for scenario: %s", state["scenario"])
    logger.info("Final counts: considerations=%d, qa_pairs=%d", len(state.get("considerations", [])), len(state.get("qa_pairs", [])))
    return FinalOutput(
        scenario=ScenarioOut(name=state["scenario"]),
        search_plan=state["search_plan"],
        considerations=state["considerations"],
        qa_pairs=state["qa_pairs"]
    )


# ---------------------------
# 6) Build graph
# ---------------------------

def build_workflow(llm, retriever):
    g = StateGraph(State)

    g.add_node("plan", lambda s: make_plan(s, llm))
    # Parallelize vectorstore queries and LLM relevance checks; tune concurrency via params below
    g.add_node("retrieve", lambda s: retrieve(s, retriever, query_parallelism=36))
    g.add_node("filter", lambda s: filter_relevance(s, llm, max_chunks=200, llm_concurrency=36))
    g.add_node("summarize", lambda s: summarize_considerations(s, llm))
    g.add_node("qa", lambda s: generate_qa(s, llm))

    g.set_entry_point("plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "filter")
    g.add_edge("filter", "summarize")
    g.add_edge("summarize", "qa")
    g.add_edge("qa", END)
    return g.compile(), assemble


base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
api_key  = os.getenv("OPENAI_API_KEY")
embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
embed_model = os.getenv("EMBEDDING_MODEL_NAME")

# LLM and vector store initializations (only run when executing the script)
llm = ChatOpenAI(model_name="Qwen/Qwen3-30B-A3B-Instruct-2507", base_url=base_url, api_key=api_key)

embeddings = OpenAIEmbeddings(base_url=embedding_base_url, model=embed_model)

vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name="rag_collection",
        connection_args={"uri": "./milvus_demo.db"},
        # ✅ Add BM25 sparse field
        builtin_function=BM25BuiltInFunction(output_field_names="sparse"),
        index_params=[
            # 1) dense vector field index
                {"index_type": "FLAT", "metric_type": "L2"},
                # sparse (Lite supports this)
                {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
            # ✅ Tell Milvus you have 2 vector fields now
                ],
        vector_field=["dense", "sparse"],
    )
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8})
# results = vectorstore.similarity_search_with_score("Barnevern", k=10,)
# 3) Workflow
app, assemble_fn = build_workflow(llm, retriever)

if __name__ == "__main__":
    # 4) Run
    state0: State = {
        "scenario": "children 8 years isnt sleeping on time",
        "search_plan": None,
        "retrieved_chunks": [],
        "candidate_chunks": [],
        "relevant_chunks": [],
        "considerations": [],
        "qa_pairs": [],
    }

    final_state = app.invoke(state0)
    output = assemble_fn(final_state)
    # Properly outputting the output (pretty JSON)
    print(output.model_dump_json(indent=2, ensure_ascii=False))

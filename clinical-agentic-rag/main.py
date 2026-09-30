"""Command-line entry point for the Clinical Agentic RAG project.

Typical first run (in order):
    python main.py ingest
    python main.py build-docs
    python main.py index --reset
    python main.py ask "What clinical trials are studying lung cancer?"
"""
import argparse

from clinical_rag import config


def cmd_ingest(args):
    from clinical_rag import ingest

    sources = set(args.sources)
    if "all" in sources:
        sources = {"clinicaltrials", "fda", "pubmed"}

    if "clinicaltrials" in sources:
        ingest.fetch_clinical_trials(args.trials)
    if "fda" in sources:
        ingest.fetch_fda_events(args.fda)
    if "pubmed" in sources:
        ingest.fetch_pubmed(args.pubmed_term, args.pubmed_max)


def cmd_build_docs(_args):
    from clinical_rag import documents as d
    from clinical_rag.ingest import load_frame

    builders = [
        ("clinical_trials", d.build_clinical_documents),
        ("fda_drug_events", d.build_fda_documents),
        ("pubmed_articles", d.build_pubmed_documents),
    ]

    docs = []
    for name, builder in builders:
        try:
            docs.extend(builder(load_frame(name)))
        except FileNotFoundError as exc:
            print(f"Skipping {name}: {exc}")

    print("Documents built:", dict(d.count_by_source(docs)))
    d.save_json(docs, "combined_clinical_documents.json")

    docs = d.clean_documents(docs)
    print("After cleaning / de-duplication:", dict(d.count_by_source(docs)))
    d.save_json(docs, "cleaned_clinical_documents.json")

    if not docs:
        print("Warning: No documents remained after cleaning! Check clean_documents logic.")
        return

    # Check which function name documents.py exposes for chunking
    chunk_fn = getattr(d, "chunk_documents", getattr(d, "chunk_all_documents", None))
    save_fn = getattr(d, "save_chunks", getattr(d, "save_chunks_to_file", None))

    if chunk_fn is None:
        raise AttributeError("clinical_rag.documents has no function for chunking documents.")

    chunks = chunk_fn(docs)
    print(f"Chunks created: {len(chunks)}", dict(d.count_by_source(chunks)))

    if save_fn:
        save_fn(chunks)
    else:
        d.save_json(chunks, "chunks.jsonl")

def cmd_index(args):
    from clinical_rag.documents import load_chunks
    from clinical_rag.indexing import build_index

    chunks = load_chunks()
    build_index(chunks, reset=args.reset, resume=args.resume)


def cmd_status(_args):
    from clinical_rag.indexing import print_status

    print_status()


def cmd_ask(args):
    from clinical_rag.generation import ask_rag

    result = ask_rag(args.question, top_k=args.top_k, show_evidence=args.show_evidence)
    print("\n" + "=" * 80 + "\nFINAL ANSWER\n" + "=" * 80)
    print(result["answer"])


def cmd_chat(args):
    from clinical_rag.generation import ask_rag

    print("Clinical Agentic RAG - type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue
        result = ask_rag(question, top_k=args.top_k, show_evidence=args.show_evidence)
        print("\n" + "-" * 80 + "\n" + result["answer"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clinical Agentic RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Download data from the three APIs")
    p.add_argument("--sources", nargs="+", default=["all"],
                   choices=["all", "clinicaltrials", "fda", "pubmed"])
    p.add_argument("--trials", type=int, default=config.N_CLINICAL_TRIALS)
    p.add_argument("--fda", type=int, default=config.N_FDA)
    p.add_argument("--pubmed-term", default=config.PUBMED_TERM)
    p.add_argument("--pubmed-max", type=int, default=config.N_PUBMED)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("build-docs", help="Build, clean and chunk documents")
    p.set_defaults(func=cmd_build_docs)

    p = sub.add_parser("index", help="Embed chunks and store them in Qdrant")
    p.add_argument("--reset", action="store_true", help="Delete existing Qdrant collection first")
    p.add_argument("--resume", action="store_true", help="Continue interrupted indexing run")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("status", help="Show Qdrant collection status and indexed point count")
    p.set_defaults(func=cmd_status)

    for name, func, helptext in [
        ("ask", cmd_ask, "Ask one question"),
        ("chat", cmd_chat, "Interactive question loop"),
    ]:
        p = sub.add_parser(name, help=helptext)
        if name == "ask":
            p.add_argument("question")
        p.add_argument("--top-k", type=int, default=config.TOP_K)
        p.add_argument("--show-evidence", action="store_true",
                       help="Print the retrieved evidence chunks")
        p.set_defaults(func=func)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
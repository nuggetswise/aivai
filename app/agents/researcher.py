from app.models import Avatar, EvidenceBundle, SourceDoc
from app.retrieval import indexer
from typing import List

# The Researcher agent is responsible for gathering evidence for a given topic and avatar.
def gather_evidence(topic: str, avatar: Avatar, links: List[str]) -> EvidenceBundle:
    """
    For a given topic and avatar, fetch, clean, chunk, embed, and index documents,
    then return an EvidenceBundle containing the sources.
    """
    # Index the corpus for this avatar (fetch, clean, chunk, embed, index)
    indexer.index_corpus(links, avatar.id)
    # For now, just create a bundle with the fetched documents (stub)
    docs = indexer.fetch_documents(links)
    bundle = EvidenceBundle(
        id=f"bundle-{avatar.id}-{topic}",
        avatar_id=avatar.id,
        topic=topic,
        sources=docs
    )
    return bundle

from clinical_rag.config import SOURCE_CLINICAL, SOURCE_FDA, SOURCE_PUBMED
from clinical_rag.retrieval import route_query


def test_fda_route():
    assert route_query("What adverse effects were reported?") == [SOURCE_FDA]


def test_clinical_route():
    assert route_query("What Phase 3 clinical trials are available?") == [SOURCE_CLINICAL]


def test_pubmed_route():
    assert route_query("What does published research say about this treatment?") == [SOURCE_PUBMED]


def test_complete_route():
    assert route_query("Give me a complete evidence summary of this treatment.") == [
        SOURCE_CLINICAL, SOURCE_FDA, SOURCE_PUBMED,
    ]


def test_default_route_is_all_sources():
    assert len(route_query("hello")) == 3

from ragpipe.workflow import build_viz_workflow


def test_build_viz_workflow_has_substrate_seam_nodes():
    wf = build_viz_workflow()
    from agent_framework import WorkflowViz

    diagram = WorkflowViz(wf).to_mermaid()
    for stage in ["retrieve", "rerank", "generate", "faithfulness", "answer"]:
        assert stage in diagram


def test_build_viz_workflow_drops_legacy_retrieval_nodes():
    wf = build_viz_workflow()
    from agent_framework import WorkflowViz

    diagram = WorkflowViz(wf).to_mermaid()
    # The fixed dense/bm25/rrf topology is gone — retrieval is one substrate call now.
    for legacy in ["dense", "bm25", "rrf"]:
        assert legacy not in diagram

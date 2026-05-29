from ragpipe.workflow import build_viz_workflow


def test_build_viz_workflow_has_all_stage_nodes():
    wf = build_viz_workflow()
    from agent_framework import WorkflowViz

    diagram = WorkflowViz(wf).to_mermaid()
    for stage in ["dense", "bm25", "rrf", "rerank", "generate", "faithfulness"]:
        assert stage in diagram

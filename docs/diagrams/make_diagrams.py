"""Render the write-up diagrams (official GCP icons via the `diagrams` library).

    pip install diagrams==0.25.1   # needs Graphviz (`dot`) on PATH
    python docs/diagrams/make_diagrams.py

Outputs PNGs into docs/images/. No data is read; every label below is architecture only.
"""
from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.gcp.compute import ComputeEngine
from diagrams.gcp.management import Quotas
from diagrams.gcp.ml import VertexAI
from diagrams.gcp.network import FirewallRules, VPC
from diagrams.gcp.security import IAP, Iam
from diagrams.gcp.storage import GCS, PersistentDisk
from diagrams.generic.storage import Storage
from diagrams.onprem.client import User
from diagrams.onprem.vcs import Github
from diagrams.programming.flowchart import Decision, Document, InputOutput, Merge, PredefinedProcess
from diagrams.programming.language import Bash, Python

OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(parents=True, exist_ok=True)

GRAPH = {"fontsize": "20", "fontname": "Helvetica", "pad": "0.4", "nodesep": "0.6", "ranksep": "0.9"}
NODE = {"fontsize": "12", "fontname": "Helvetica"}


def cloud_architecture() -> None:
    with Diagram("Cloud architecture: Codespace -> Google Cloud (us-central1)", filename=str(OUT / "cloud_architecture"),
                 show=False, direction="LR", graph_attr={**GRAPH, "splines": "spline"}, node_attr=NODE):
        dev = User("Developer")
        with Cluster("GitHub Codespace (devcontainer, Python 3.12)"):
            make = Bash("Makefile +\ngcp_cluster.sh")
            local = Python("train.py\n(local / fast_config)")
            agent = Python("ADK agent\nsasol_analyst")
        repo = Github("GitHub repo\n(code only)")

        with Cluster("GCP project"):
            iam = Iam("IAM\n(gcloud auth)")
            quotas = Quotas("Quotas\nGPU = 0, CPUS_ALL_REGIONS = 12")
            vertex = VertexAI("Vertex AI\nGemini (agent LLM)")
            bucket = GCS("Cloud Storage bucket\ndata/raw | processed cache\nsubmissions/<RUN_ID> | sessions")
            with Cluster("VPC sasol-net (custom subnet)"):
                fw = FirewallRules("Firewall\ntcp:22 from\n35.235.240.0/20 only")
                iap = IAP("Identity-Aware Proxy\nSSH tunnel")
                with Cluster("Compute Engine (Spot, stop on preempt)"):
                    vm = ComputeEngine("n2-custom-12-24576\nDebian 12, uv venv")
                    disk = PersistentDisk("pd-balanced\n100 GB boot")

        dev >> make
        dev >> agent
        make >> Edge(label="tar | ssh\n(code sync)") >> iap >> fw >> vm
        make >> Edge(label="rsync raw data\nfetch results", style="dashed") >> bucket
        vm - disk
        vm >> Edge(label="read raw + cache\nwrite submissions") >> bucket
        iam - quotas
        make >> Edge(style="dotted") >> iam
        agent >> Edge(label="Gemini calls") >> vertex
        agent >> Edge(label="audit tools (read-only)", style="dotted") >> bucket
        local >> Edge(style="dashed", label="push code") >> repo


def run_lifecycle() -> None:
    with Diagram("Run lifecycle: one `make gcp-run-cpu`", filename=str(OUT / "run_lifecycle"), show=False,
                 direction="LR", graph_attr=GRAPH, node_attr=NODE):
        init = Bash("gcp-init\nenable APIs, VPC,\nIAP firewall, bucket")
        up = Decision("gcp-up\nstart existing VM or\nprovision: machine x zone\nfallback loop")
        vm = ComputeEngine("Spot VM\nRUNNING")
        sync = Bash("gcp-sync-code\ntar over IAP SSH")
        boot = Python("remote_bootstrap.sh\nuv venv, skipped when\nrequirements hash matches")
        train = Python("remote_train.sh\nsetsid + nohup:\nsurvives SSH drops")
        bucket = GCS("gs://.../submissions/\n<RUN_ID>/")
        fetch = Document("gcp-fetch\ndownload + re-validate\nsubmission contract")
        down = ComputeEngine("gcp-down\nSTOP: vCPU billing ends,\ndisk kept")
        init >> up >> vm >> sync >> boot >> train
        train >> Edge(label="validated CSV +\noof_metrics.json") >> bucket >> fetch >> down


def ml_pipeline() -> None:
    with Diagram("Modelling pipeline: multi-cohort, three-head OOF stack", filename=str(OUT / "ml_pipeline"),
                 show=False, direction="LR", graph_attr={**GRAPH, "splines": "spline", "ranksep": "1.1"},
                 node_attr=NODE):
        raw = Storage("Transactions\n(line items, private)")
        rules = PredefinedProcess("organiser\nlabel_rules.py")
        cohorts = InputOutput("Cohort builder\n17 month-start cutoffs\nX: ts < cutoff\ny: [cutoff, +3 mo)")
        with Cluster("Features (Polars, f4: ~300-380 cols)"):
            rfm = Python("RFM, momentum,\ncadence, per-category")
            replay = Python("rule-replay lags")
            emb = Python("item-text embeddings\nSVD(16) + affinities")
        X = Merge("feature matrix\nGroupKFold(ID) x5")
        with Cluster("Head 1: CLV_fuel"):
            h1 = Python("LGBM / CatBoost / XGB\n+ Huber LGBM\nsimplex blend")
        with Cluster("Head 2: CLV_nonfuel"):
            h2 = Python("direct GBDTs +\nhurdle P(z>0) x E[z|z>0]\nsimplex blend")
        with Cluster("Head 3: Opportunity (17 classes)"):
            h3 = Python("GBDTs on X +\nOOF meta-features")
            mult = Decision("nested-OOF\nclass multipliers")
        sub = Document("submission.py\ncontract checks")

        [raw, rules] >> cohorts >> [rfm, replay, emb]
        [rfm, replay, emb] >> X
        X >> [h1, h2, h3]
        h1 >> Edge(label="OOF z_fuel", style="dashed") >> h3
        h2 >> Edge(label="OOF z_nonfuel, P(z>0)", style="dashed") >> h3
        h3 >> mult
        [h1, h2, mult] >> sub


if __name__ == "__main__":
    cloud_architecture()
    run_lifecycle()
    ml_pipeline()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))

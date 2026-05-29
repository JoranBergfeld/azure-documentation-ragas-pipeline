from ragpipe.corpus import select_urls

BASE = "https://learn.microsoft.com/en-us/azure"


def test_groups_by_service_and_caps_per_service():
    urls = [
        f"{BASE}/cosmos-db/overview",
        f"{BASE}/cosmos-db/concepts-x",
        f"{BASE}/cosmos-db/how-to-y",
        f"{BASE}/cosmos-db/tutorial-z",
        f"{BASE}/storage/overview",
        f"{BASE}/storage/quickstart-a",
    ]
    out = select_urls(urls, per_service=2, cap=100)
    cosmos = [u for u in out if "/cosmos-db/" in u]
    storage = [u for u in out if "/storage/" in u]
    assert len(cosmos) == 2
    assert len(storage) == 2


def test_prefers_overview_and_concept_pages():
    urls = [
        f"{BASE}/svc/tutorial-deep",
        f"{BASE}/svc/how-to-thing",
        f"{BASE}/svc/overview",
        f"{BASE}/svc/concepts-model",
    ]
    out = select_urls(urls, per_service=2, cap=100)
    assert out[0].endswith("/overview")
    assert "concepts-model" in out[1]


def test_round_robin_spreads_across_services_under_cap():
    urls = []
    for svc in ("a", "b", "c"):
        for i in range(5):
            urls.append(f"{BASE}/{svc}/overview-{i}")
    # cap below total: every service should still get its first pick before
    # any service gets a second.
    out = select_urls(urls, per_service=5, cap=3)
    services = {u.split("/azure/")[1].split("/")[0] for u in out}
    assert services == {"a", "b", "c"}
    assert len(out) == 3


def test_drops_media_and_non_azure_noise():
    urls = [
        f"{BASE}/cosmos-db/media/diagram.png",
        f"{BASE}/cosmos-db/includes/snippet",
        f"{BASE}/cosmos-db/overview",
        "https://learn.microsoft.com/en-us/dotnet/overview",  # not /azure/
    ]
    out = select_urls(urls, per_service=10, cap=100)
    assert out == [f"{BASE}/cosmos-db/overview"]


def test_global_cap_enforced():
    urls = [f"{BASE}/svc{i}/overview" for i in range(50)]
    out = select_urls(urls, per_service=4, cap=10)
    assert len(out) == 10

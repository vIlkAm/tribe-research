"""Offline tests for tools/runpod.py. A fake transport replaces `_http`; nothing
here can reach the network (the autouse fixture makes any real call fail)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import runpod  # noqa: E402

REAL_HTTP = runpod._http  # captured before the fixture swaps in the fake

API_KEY = "rpa_TESTKEY_0123456789abcdef"
HF_TOKEN = "hf_TESTTOKEN_abcdefghijklmnop"
PUBKEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITEST tyler@server"
T0 = 1_790_000_000.0  # fixed clock


class FakeApi:
    """Routes (method, url-without-query) to canned (status, json) replies; records calls."""

    def __init__(self):
        self.routes = {}
        self.calls = []

    def on(self, method, path, status=200, body=None, graphql_key=None):
        self.routes[(method, path, graphql_key)] = (status, body)

    def __call__(self, method, url, body=None):
        self.calls.append((method, url, body))
        path = url.split("?")[0].replace(runpod.REST_URL, "")
        key = None
        if url == runpod.GRAPHQL_URL:
            path = "graphql"
            q = body["query"]
            key = next((k for (_, p, k) in self.routes if p == "graphql" and k and k in q), None)
        if (method, path, key) not in self.routes:
            raise AssertionError(f"unexpected call {method} {url}")
        status, reply = self.routes[(method, path, key)]
        if callable(reply):
            reply = reply(body)
        raw = b"" if reply is None else (reply if isinstance(reply, bytes) else json.dumps(reply).encode())
        return status, raw


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f"# test\nRUNPOD_API_KEY={API_KEY}\nexport HF_TOKEN=\"{HF_TOKEN}\"\n\nSHORT=x\n")
    pub = tmp_path / "id_ed25519.pub"
    pub.write_text(PUBKEY + "\n")
    monkeypatch.setattr(runpod, "ENV_FILE", env)
    monkeypatch.setattr(runpod, "STATE_FILE", tmp_path / "results" / "runpod_state.json")
    monkeypatch.setattr(runpod, "HEARTBEAT_FILE", tmp_path / "results" / "runpod_watchdog.json")
    monkeypatch.setattr(runpod, "PUBKEY_FILE", pub)
    monkeypatch.setattr(runpod, "_ENV_CACHE", None)
    monkeypatch.setattr(runpod, "_SECRETS", set())
    monkeypatch.setattr(runpod, "_sleep", lambda s: None)

    def no_network(*a, **k):
        raise AssertionError("real network call attempted")

    monkeypatch.setattr(runpod.urllib.request, "urlopen", no_network)
    fake = FakeApi()
    monkeypatch.setattr(runpod, "_http", fake)
    return fake


def run(argv, capsys):
    code = runpod.main(argv)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def assert_no_secrets(*texts):
    for t in texts:
        assert HF_TOKEN not in t
        assert API_KEY not in t


# --------------------------------------------------------------------------- env + redaction

def test_parse_env_handles_comments_export_and_quotes():
    env = runpod.parse_env("# c\n\nA=1\nexport B = 'two words'\nC=\"q\"\nnoequals\nD=a=b\n")
    assert env == {"A": "1", "B": "two words", "C": "q", "D": "a=b"}


def test_load_env_registers_only_credential_length_values():
    env = runpod.load_env()
    assert env["HF_TOKEN"] == HF_TOKEN and env["RUNPOD_API_KEY"] == API_KEY
    assert runpod.scrub(f"a {HF_TOKEN} b {API_KEY} x") == "a *** b *** x"  # "x" (SHORT) untouched


def test_need_missing_key_is_usage_error(tmp_path, monkeypatch):
    empty = tmp_path / "empty.env"
    empty.write_text("")
    monkeypatch.setattr(runpod, "ENV_FILE", empty)
    with pytest.raises(runpod.UsageError):
        runpod.need("HF_TOKEN")


def test_redact_masks_env_in_rest_and_graphql_shapes():
    runpod.load_env()
    pod = {"id": "p1", "env": {"HF_TOKEN": HF_TOKEN, "PUBLIC_KEY": PUBKEY},
           "nested": [{"env": [f"HF_TOKEN={HF_TOKEN}"], "apiKey": "k" * 20}],
           "lastStatusChange": f"echo {HF_TOKEN}"}
    r = runpod.redact(pod)
    assert r["env"] == {"HF_TOKEN": "***", "PUBLIC_KEY": "***"}
    assert r["nested"][0]["env"] == ["HF_TOKEN=***"]
    assert r["nested"][0]["apiKey"] == "***"
    assert_no_secrets(json.dumps(r))


def test_pod_list_json_never_shows_token(isolated, capsys):
    isolated.on("GET", "/pods", body=[{"id": "p1", "name": "tribe-a", "desiredStatus": "RUNNING",
                                      "costPerHr": "0.86", "env": {"HF_TOKEN": HF_TOKEN},
                                      "publicIp": "1.2.3.4", "portMappings": {"22": 17445}}])
    code, o, e = run(["pod-list", "--json"], capsys)
    assert code == 0 and '"HF_TOKEN": "***"' in o
    code, o2, e2 = run(["pod-list"], capsys)
    assert code == 0 and "root@1.2.3.4 -p 17445" in o2
    assert_no_secrets(o, e, o2, e2)


def test_api_error_echoing_request_is_scrubbed(isolated, capsys):
    isolated.on("POST", "/pods", status=400,
                body=lambda b: {"error": "bad input", "echo": b, "auth": f"Bearer {API_KEY}"})
    runpod.write_json_atomic(runpod.HEARTBEAT_FILE, {"epoch": 9e12, "interval": 60})
    code, o, e = run(["pod-create", "--name", "tribe-x", "--gpu-type", "L40S", "--max-hours", "2"], capsys)
    assert code == 1 and "HTTP 400" in e and "pod-list" in e
    assert_no_secrets(o, e)
    assert len([c for c in isolated.calls if c[0] == "POST"]) == 1  # never retried


# --------------------------------------------------------------------------- pod-create

def test_pod_create_cheap_default_is_community_on_demand_with_pod_volume(isolated, capsys, monkeypatch):
    monkeypatch.setattr(runpod, "now_ts", lambda: T0)
    isolated.on("POST", "/pods", status=201, body={"id": "podc", "costPerHr": "0.34"})
    runpod.write_json_atomic(runpod.HEARTBEAT_FILE, {"epoch": T0 - 30, "interval": 60})
    code, o, e = run(["pod-create", "--name", "tribe-c", "--gpu-type", "4090", "--max-hours", "3"], capsys)
    assert code == 0, e
    assert [c[0] for c in isolated.calls] == ["POST"]            # no volume lookup
    body = isolated.calls[0][2]
    assert body["cloudType"] == "COMMUNITY" and body["interruptible"] is False
    assert body["volumeInGb"] == 120 and body["volumeMountPath"] == "/workspace"
    assert "networkVolumeId" not in body and "dataCenterIds" not in body
    assert body["containerDiskInGb"] == 30 and body["minRAMPerGPU"] == 48
    assert body["ports"] == ["22/tcp"] and body["supportPublicIp"] is True
    assert body["env"] == {"PUBLIC_KEY": PUBKEY, "HF_TOKEN": HF_TOKEN}
    assert "DELETES" in o                                          # pod-volume warning
    state = json.loads(runpod.STATE_FILE.read_text())["pods"]["podc"]
    assert state["cloud"] == "COMMUNITY" and state["pod_volume_gb"] == 120 and state["volume_id"] is None
    assert_no_secrets(o, e)


def test_pod_create_spot_only_when_flagged_and_volume_needs_secure(isolated, capsys):
    code, o, e = run(["pod-create", "--name", "tribe-s", "--gpu-type", "3090", "--max-hours", "1",
                      "--interruptible", "--volume-gb", "80", "--min-ram-gb", "60", "--dry-run"], capsys)
    assert code == 0
    body = json.loads(o[:o.rindex("}") + 1])
    assert body["interruptible"] is True and body["volumeInGb"] == 80 and body["minRAMPerGPU"] == 60
    code, _, e = run(["pod-create", "--name", "tribe-s", "--gpu-type", "3090", "--max-hours", "1",
                      "--volume-id", "v", "--dry-run"], capsys)
    assert code == 2 and "Secure Cloud" in e
    assert isolated.calls == []


def test_pod_create_payload_shape_and_state(isolated, capsys, monkeypatch):
    monkeypatch.setattr(runpod, "now_ts", lambda: T0)
    isolated.on("GET", "/networkvolumes/vol1", body={"id": "vol1", "dataCenterId": "EU-RO-1", "size": 150})
    isolated.on("POST", "/pods", status=201, body={"id": "pod123", "costPerHr": "1.72",
                                                   "env": {"HF_TOKEN": HF_TOKEN}})
    runpod.write_json_atomic(runpod.HEARTBEAT_FILE, {"epoch": T0 - 30, "interval": 60,
                                                     "max_usd": 40, "max_hours": 8})
    code, o, e = run(["pod-create", "--name", "tribe-p2", "--gpu-type", "L40S", "--gpu-count", "2",
                      "--cloud", "SECURE", "--volume-id", "vol1", "--max-hours", "3"], capsys)
    assert code == 0, e
    body = next(c[2] for c in isolated.calls if c[0] == "POST")
    assert body["ports"] == ["22/tcp"]
    assert body["supportPublicIp"] is True
    assert body["networkVolumeId"] == "vol1" and body["volumeMountPath"] == "/workspace"
    assert body["dataCenterIds"] == ["EU-RO-1"] and "volumeInGb" not in body
    assert body["env"] == {"PUBLIC_KEY": PUBKEY, "HF_TOKEN": HF_TOKEN}
    assert body["cloudType"] == "SECURE" and body["interruptible"] is False
    assert body["gpuTypeIds"] == ["NVIDIA L40S"] and body["gpuCount"] == 2
    assert body["imageName"] == runpod.DEFAULT_IMAGE
    assert "12.8" in body["allowedCudaVersions"] and "12.4" not in body["allowedCudaVersions"]
    state = json.loads(runpod.STATE_FILE.read_text())["pods"]["pod123"]
    assert state["max_hours"] == 3 and state["cost_per_hr"] == 1.72 and state["datacenter"] == "EU-RO-1"
    assert_no_secrets(o, e, runpod.STATE_FILE.read_text())


def test_pod_create_dry_run_redacts_and_sends_nothing(isolated, capsys):
    code, o, e = run(["pod-create", "--name", "tribe-d", "--gpu-type", "4090", "--cloud", "SECURE",
                      "--volume-id", "v", "--dc", "US-TX-3", "--max-hours", "1", "--dry-run"], capsys)
    assert code == 0 and isolated.calls == []
    assert '"HF_TOKEN": "***"' in o and "NVIDIA GeForce RTX 4090" in o
    assert_no_secrets(o, e)


def test_pod_create_refuses_without_watchdog_and_bad_names(isolated, capsys):
    base = ["pod-create", "--gpu-type", "L40S", "--max-hours", "1"]
    code, _, e = run(base + ["--name", "tribe-a"], capsys)
    assert code == 2 and "watchdog" in e
    code, _, e = run(base + ["--name", "other-a"], capsys)
    assert code == 2 and "tribe-" in e
    code, _, e = run(["pod-create", "--name", "tribe-b", "--gpu-type", "B200",
                      "--max-hours", "1", "--dry-run"], capsys)
    assert code == 2 and "Blackwell" in e
    assert isolated.calls == []


def test_resolve_gpu():
    assert runpod.resolve_gpu("L40S") == "NVIDIA L40S"
    assert runpod.resolve_gpu("l40") == "NVIDIA L40"
    assert runpod.resolve_gpu("RTX 4090") == "NVIDIA GeForce RTX 4090"
    assert runpod.resolve_gpu("A6000") == "NVIDIA RTX A6000"
    assert runpod.resolve_gpu("H100 80GB HBM3") == "NVIDIA H100 80GB HBM3"
    with pytest.raises(runpod.UsageError, match="ambiguous"):
        runpod.resolve_gpu("H100")
    assert not runpod.is_blackwell("NVIDIA RTX 5000 Ada Generation")
    assert runpod.is_blackwell("NVIDIA GeForce RTX 5090")


# --------------------------------------------------------------------------- pod-wait / ssh

def test_ssh_endpoint_extraction():
    assert runpod.ssh_endpoint({"publicIp": "213.173.108.12", "portMappings": {"22": 17445}}) == ("213.173.108.12", 17445)
    assert runpod.ssh_endpoint({"publicIp": "1.1.1.1", "portMappings": {22: 1}}) == ("1.1.1.1", 1)
    assert runpod.ssh_endpoint({"publicIp": "", "portMappings": {"22": 1}}) is None
    assert runpod.ssh_endpoint({"publicIp": "1.1.1.1", "portMappings": {"8888": 1}}) is None
    assert runpod.ssh_endpoint({"publicIp": "1.1.1.1"}) is None


def test_pod_wait_polls_until_ssh_and_prints_only_export(isolated, capsys):
    replies = iter([
        {"id": "p", "desiredStatus": "RUNNING", "publicIp": ""},
        {"id": "p", "desiredStatus": "RUNNING", "publicIp": "9.9.9.9", "portMappings": {}},
        {"id": "p", "desiredStatus": "RUNNING", "publicIp": "9.9.9.9", "portMappings": {"22": 40022}},
    ])
    isolated.on("GET", "/pods/p", body=lambda b: next(replies))
    isolated.on("POST", "graphql", graphql_key="runtime { ports",
                body={"data": {"myself": {"pods": [{"id": "p", "runtime": None}]}}})
    code, o, e = run(["pod-wait", "p"], capsys)
    assert code == 0
    assert o == "export POD=root@9.9.9.9 POD_PORT=40022\n"


def test_pod_wait_fails_on_terminated(isolated, capsys):
    isolated.on("GET", "/pods/p", body={"id": "p", "desiredStatus": "EXITED",
                                        "lastStatusChange": "Terminated by user: x"})
    code, o, e = run(["pod-wait", "p"], capsys)
    assert code == 1 and o == ""


# --------------------------------------------------------------------------- watchdog

def pod(pid, name, rate, created=None, status="RUNNING"):
    p = {"id": pid, "name": name, "costPerHr": str(rate), "desiredStatus": status}
    if created is not None:
        p["createdAt"] = runpod.iso(created)
    return p


def test_watchdog_max_hours_uses_lower_of_pod_and_global_cap():
    wd = runpod.Watchdog(max_usd=1000, max_hours=8)
    state = {"pods": {"a": {"created_at": runpod.iso(T0 - 2.5 * 3600), "max_hours": 2},
                      "b": {"created_at": runpod.iso(T0 - 2.5 * 3600), "max_hours": 10}}}
    pods = [pod("a", "tribe-a", 1.0), pod("b", "tribe-b", 1.0), pod("c", "tribe-c", 1.0, created=T0 - 9 * 3600)]
    line, kills = wd.tick(pods, None, state, T0)
    assert [k for k, _, _ in kills] == ["a", "c"]
    reasons = {k: r for k, _, r in kills}
    assert "max-hours 2" in reasons["a"] and "max-hours 8" in reasons["c"]


def test_watchdog_start_is_earliest_of_state_and_api():
    wd = runpod.Watchdog(max_usd=1000, max_hours=3)
    state = {"pods": {"a": {"created_at": runpod.iso(T0 - 600)}}}
    _, kills = wd.tick([pod("a", "tribe-a", 1.0, created=T0 - 4 * 3600)], None, state, T0)
    assert [k for k, _, _ in kills] == ["a"]


def test_watchdog_max_usd_terminates_all_tribe_pods_only():
    wd = runpod.Watchdog(max_usd=10, max_hours=100)
    pods = [pod("a", "tribe-a", 2.0, created=T0 - 3 * 3600),   # $6
            pod("b", "tribe-b", 2.0, created=T0 - 2 * 3600),   # $4 -> total $10
            pod("x", "someone-else", 50.0, created=T0 - 10 * 3600)]
    line, kills = wd.tick(pods, None, {"pods": {}}, T0)
    assert sorted(k for k, _, _ in kills) == ["a", "b"]
    assert all("max-usd" in r for _, _, r in kills)
    assert "est=$10.00" in line


def test_watchdog_under_budget_kills_nothing_and_counts_ended_pods():
    wd = runpod.Watchdog(max_usd=10, max_hours=100)
    _, kills = wd.tick([pod("a", "tribe-a", 2.0, created=T0 - 3 * 3600)], None, {"pods": {}}, T0)
    assert kills == []                                               # $6 < $10
    # pod a disappears (terminated elsewhere): its $6 stays in this run's total
    _, kills = wd.tick([pod("b", "tribe-b", 2.0, created=T0 - 2 * 3600)], None, {"pods": {}}, T0 + 1)
    assert [k for k, _, _ in kills] == ["b"]


def test_watchdog_list_flicker_does_not_double_count():
    wd = runpod.Watchdog(max_usd=10, max_hours=100)
    a = [pod("a", "tribe-a", 2.0, created=T0 - 3 * 3600)]            # $6: over one copy, under two
    assert wd.tick(a, None, {"pods": {}}, T0)[1] == []
    assert wd.tick([], None, {"pods": {}}, T0 + 1)[1] == []          # a missing for one tick
    line, kills = wd.tick(a, None, {"pods": {}}, T0 + 2)
    assert kills == [] and "est=$6.00" in line


def test_watchdog_ignores_terminated_pods():
    wd = runpod.Watchdog(max_usd=1, max_hours=1)
    dead = pod("a", "tribe-a", 5.0, created=T0 - 9 * 3600, status="EXITED")
    dead["lastStatusChange"] = "Terminated by user: x"
    _, kills = wd.tick([dead], None, {"pods": {}}, T0)
    assert kills == []


def gpu_rt(*utils):
    return {"uptimeInSeconds": 100, "gpus": [{"id": f"g{i}", "gpuUtilPercent": u} for i, u in enumerate(utils)]}


def test_watchdog_idle_only_after_gpu_was_used():
    wd = runpod.Watchdog(max_usd=1000, max_hours=100, idle_minutes=20)
    p = [pod("a", "tribe-a", 1.0, created=T0)]
    # setup phase: 0% for an hour, never armed -> never killed
    for m in range(0, 61, 5):
        _, kills = wd.tick(p, {"a": gpu_rt(0)}, {"pods": {}}, T0 + m * 60)
        assert kills == []
    t = T0 + 3600
    wd.tick(p, {"a": gpu_rt(95, 0)}, {"pods": {}}, t)               # busy -> armed
    _, kills = wd.tick(p, {"a": gpu_rt(0, 0)}, {"pods": {}}, t + 60)  # idle clock starts
    assert kills == []
    _, kills = wd.tick(p, None, {"pods": {}}, t + 10 * 60)         # telemetry down: no change
    assert kills == []
    _, kills = wd.tick(p, {"a": {"gpus": None}}, {"pods": {}}, t + 15 * 60)
    assert kills == []
    _, kills = wd.tick(p, {"a": gpu_rt(0, 0)}, {"pods": {}}, t + 60 + 20 * 60)
    assert [k for k, _, _ in kills] == ["a"] and "idle" in kills[0][2]


def test_watchdog_busy_resets_idle_and_zero_disables():
    wd = runpod.Watchdog(max_usd=1000, max_hours=100, idle_minutes=20)
    p = [pod("a", "tribe-a", 1.0, created=T0)]
    wd.tick(p, {"a": gpu_rt(50)}, {"pods": {}}, T0)
    wd.tick(p, {"a": gpu_rt(0)}, {"pods": {}}, T0 + 60)
    wd.tick(p, {"a": gpu_rt(10)}, {"pods": {}}, T0 + 15 * 60)
    _, kills = wd.tick(p, {"a": gpu_rt(0)}, {"pods": {}}, T0 + 30 * 60)
    assert kills == []
    off = runpod.Watchdog(max_usd=1000, max_hours=100, idle_minutes=0)
    off.tick(p, {"a": gpu_rt(50)}, {"pods": {}}, T0)
    off.tick(p, {"a": gpu_rt(0)}, {"pods": {}}, T0 + 60)
    _, kills = off.tick(p, {"a": gpu_rt(0)}, {"pods": {}}, T0 + 5 * 3600)
    assert kills == []


def test_watchdog_loop_terminates_retries_failed_delete_and_writes_heartbeat(isolated, capsys, monkeypatch):
    clock = iter([T0, T0 + 60])
    monkeypatch.setattr(runpod, "now_ts", lambda: next(clock))
    runpod.write_json_atomic(runpod.STATE_FILE, {"pods": {"a": {"created_at": runpod.iso(T0 - 5 * 3600),
                                                                "max_hours": 1, "volume_id": "vol1"}}})
    isolated.on("GET", "/pods", body=[pod("a", "tribe-a", 1.0), pod("z", "other", 9.0, created=T0 - 99 * 3600)])
    isolated.on("POST", "graphql", graphql_key="myself",
                body={"data": {"myself": {"clientBalance": 50, "currentSpendPerHr": 1.0,
                                          "pods": [{"id": "a", "runtime": None}]}}})
    real_call = isolated.__call__
    delete_status = iter([500, 204])  # first DELETE fails, the next one succeeds

    def fake(method, url, body=None):
        if method == "DELETE":
            isolated.calls.append((method, url, body))
            return next(delete_status), b""
        return real_call(method, url, body)

    monkeypatch.setattr(runpod, "_http", fake)
    code, o, e = run(["watchdog", "--max-usd", "100", "--max-hours", "8", "--once"], capsys)
    assert code == 0 and "TERMINATE FAILED a" in o
    assert runpod.HEARTBEAT_FILE.exists()
    code, o, e = run(["watchdog", "--max-usd", "100", "--max-hours", "8", "--once"], capsys)
    # --once starts a fresh Watchdog, so the retry comes from the cap itself here
    assert "TERMINATED a" in o
    assert all("/pods/z" not in c[1] for c in isolated.calls if c[0] == "DELETE")
    assert json.loads(runpod.STATE_FILE.read_text())["pods"]["a"]["ended_at"]
    assert_no_secrets(o, e)


def test_watchdog_pending_kill_is_retried_by_tick():
    wd = runpod.Watchdog(max_usd=1000, max_hours=100)
    p = [pod("a", "tribe-a", 1.0, created=T0)]
    wd.pending["a"] = "max-usd $5 reached"
    _, kills = wd.tick(p, None, {"pods": {}}, T0)
    assert kills == [("a", "stop", "max-usd $5 reached [retry]")]
    _, kills = wd.tick([], None, {"pods": {}}, T0 + 60)   # gone -> pending cleared
    assert kills == [] and wd.pending == {}


def test_watchdog_list_failure_is_not_zero_pods(isolated, capsys):
    isolated.on("GET", "/pods", status=503, body={"error": "down"})
    code, o, e = run(["watchdog", "--max-usd", "5", "--max-hours", "1", "--once"], capsys)
    assert code == 0 and "ERROR listing pods" in o and "no decision" in o
    assert not runpod.HEARTBEAT_FILE.exists()


# --------------------------------------------------------------------------- gpus / balance

TYPES = {"data": {"gpuTypes": [
    {"id": "NVIDIA GeForce RTX 4090", "memoryInGb": 24, "secureCloud": True, "communityCloud": True,
     "securePrice": 0.69, "communityPrice": 0.34, "secureSpotPrice": 0.5, "communitySpotPrice": 0.2},
    {"id": "NVIDIA L40S", "memoryInGb": 48, "secureCloud": True, "communityCloud": True,
     "securePrice": 0.86, "communityPrice": 0.79},
    {"id": "NVIDIA RTX A4000", "memoryInGb": 16, "secureCloud": True, "communityCloud": True,
     "securePrice": 0.17, "communityPrice": 0.09},
    {"id": "NVIDIA H100 NVL", "memoryInGb": 94, "secureCloud": True, "securePrice": 2.59}]}}
DCS = {"data": {"dataCenters": [
    {"id": "EU-RO-1", "gpuAvailability": [{"gpuTypeId": "NVIDIA L40S", "stockStatus": "High"},
                                          {"gpuTypeId": "NVIDIA RTX A4000", "stockStatus": "High"}]},
    {"id": "US-TX-3", "gpuAvailability": [{"gpuTypeId": "NVIDIA GeForce RTX 4090", "stockStatus": "Low"}]}]}}


def test_gpus_lists_cheapest_first_with_both_clouds_and_spot(isolated, capsys):
    isolated.on("POST", "graphql", graphql_key="secureSpotPrice", body=TYPES)
    isolated.on("POST", "graphql", graphql_key="dataCenters { id name", body=DCS)
    isolated.on("POST", "graphql", graphql_key="PriceDetail", body={"errors": [{"message": "Cannot query field"}]})
    isolated.on("GET", runpod.CATALOG_URL, status=404, body={"error": "not found"})
    code, o, e = run(["gpus"], capsys)
    assert code == 0, e
    rows = [ln for ln in o.splitlines()[2:-1]]
    assert rows[0].startswith("RTX 4090") and rows[1].startswith("L40S")   # $0.34 before $0.79
    assert all(x in rows[0] for x in ("0.34", "0.20", "0.69", "0.50", "US-TX-3:Low"))
    assert "A4000" not in o                                    # 16 GB VRAM: hidden
    assert "H100" not in o                                     # no stock, not on the shortlist
    assert "showing list prices" in e and "v2 catalog unavailable" in e
    code, o, e = run(["gpus", "--all"], capsys)
    assert "RTX A4000" in o and "VRAM too small" in o and "H100 NVL" in o


def test_gpus_uses_host_ram_filter_and_falls_back_without_spot(isolated, capsys):
    isolated.on("POST", "graphql", graphql_key="secureSpotPrice", body={"errors": [{"message": "no spot"}]})
    isolated.on("POST", "graphql", graphql_key="securePrice communityPrice } }", body=TYPES)
    isolated.on("POST", "graphql", graphql_key="dataCenters { id name", body=DCS)
    isolated.on("POST", "graphql", graphql_key="PriceDetail", body={"data": {
        "dataCenters": [{"id": "EU-RO-1", "storageSupport": True}],
        "gpuTypes": [{"id": "NVIDIA L40S",
                      "sec": {"stockStatus": "High", "uninterruptablePrice": 1.98, "minimumBidPrice": 1.0,
                              "minMemory": 62, "maxUnreservedGpuCount": 6},
                      "com": {"stockStatus": "Low", "uninterruptablePrice": 1.42, "minimumBidPrice": 0.6,
                              "minMemory": 100, "maxUnreservedGpuCount": 2}}]}})
    isolated.on("GET", runpod.CATALOG_URL, body={"data": [
        {"id": "NVIDIA L40S", "price": {"secure": 0.86, "community": 0.79}, "availability": "MEDIUM"},
        {"id": "NVIDIA GeForce RTX 4090", "price": {"secure": 0.69, "community": 0.31}, "availability": "HIGH"}]})
    code, o, e = run(["gpus", "--dc", "EU-RO-1", "--gpu-count", "2", "--min-ram-gb", "40"], capsys)
    assert code == 0, e
    assert "spot prices unavailable" in e
    row = next(ln for ln in o.splitlines() if ln.startswith("L40S"))
    assert "0.71" in row and "0.99" in row and "Low/High" in row and " 62 " in row and "   6 " in row
    assert "network volumes here: yes" in o and "80 GB asked" in o
    assert "MEDIUM" in row and "0.71" in row   # host-matched pod price (1.42 for 2 GPUs) shown per GPU
    r4090 = next(ln for ln in o.splitlines() if ln.startswith("RTX 4090"))
    assert "0.31" in r4090 and "HIGH" in r4090               # documented catalog price beats the old list price
    cat = next(c[1] for c in isolated.calls if c[1].startswith(runpod.CATALOG_URL))
    assert "count=2" in cat and "product=POD" in cat and "cloud=COMMUNITY" in cat
    detail = [c[2] for c in isolated.calls if c[2] and "PriceDetail" in c[2]["query"]]
    assert detail[0]["variables"] == {"n": 2, "ram": 80, "dc": "EU-RO-1"}   # GraphQL variables


def test_balance(isolated, capsys):
    isolated.on("POST", "graphql", graphql_key="spendLimit",
                body={"data": {"myself": {"clientBalance": 25.0, "currentSpendPerHr": 2.5, "spendLimit": 80}}})
    code, o, e = run(["balance"], capsys)
    assert code == 0 and "$25.00" in o and "runway 10.0 h" in o


# --------------------------------------------------------------------------- stop / terminate / pod volumes

def test_watchdog_stops_pod_volume_pods_and_terminates_network_volume_pods():
    wd = runpod.Watchdog(max_usd=1, max_hours=100)
    state = {"pods": {"n": {"volume_id": "vol1"}}}
    pods = [pod("p", "tribe-p", 1.0, created=T0 - 3600), pod("n", "tribe-n", 1.0, created=T0 - 3600),
            dict(pod("r", "tribe-r", 1.0, created=T0 - 3600), networkVolumeId="vol2")]
    _, kills = wd.tick(pods, None, state, T0)
    assert [(k, a) for k, a, _ in kills] == [("n", "terminate"), ("p", "stop"), ("r", "terminate")]


def test_watchdog_stopped_pod_is_frozen_and_left_alone():
    wd = runpod.Watchdog(max_usd=5, max_hours=2)
    p = pod("a", "tribe-a", 1.0, created=T0 - 3 * 3600)
    _, kills = wd.tick([p], None, {"pods": {}}, T0)
    assert [(k, a) for k, a, _ in kills] == [("a", "stop")]         # over max-hours
    stopped = dict(p, desiredStatus="EXITED", lastStatusChange="Exited by user: x")
    line, kills = wd.tick([stopped], None, {"pods": {}}, T0 + 10 * 3600)
    assert kills == [] and "est=$3.00" in line and "stopped" in line    # no accrual while stopped
    # a stopped pod never seen running (watchdog restarted) counts $0 and is not acted on
    fresh = runpod.Watchdog(max_usd=5, max_hours=2)
    line, kills = fresh.tick([stopped], None, {"pods": {}}, T0)
    assert kills == [] and "est=$0.00" in line


def test_watchdog_grace_after_pod_start():
    wd = runpod.Watchdog(max_usd=1, max_hours=1)
    p = [pod("a", "tribe-a", 1.0, created=T0 - 5 * 3600)]
    state = {"pods": {"a": {"grace_until": runpod.iso(T0 + 1800)}}}
    line, kills = wd.tick(p, None, state, T0)
    assert kills == [] and "grace" in line
    _, kills = wd.tick(p, None, state, T0 + 1801)
    assert [k for k, _, _ in kills] == ["a"]


def test_pod_start_records_grace(isolated, capsys, monkeypatch):
    monkeypatch.setattr(runpod, "now_ts", lambda: T0)
    isolated.on("POST", "/pods/a/start", body={"id": "a"})
    code, o, e = run(["pod-start", "a", "--grace-min", "15"], capsys)
    assert code == 0
    assert json.loads(runpod.STATE_FILE.read_text())["pods"]["a"]["grace_until"] == runpod.iso(T0 + 900)


def test_pod_terminate_pod_volume_needs_yes(isolated, capsys):
    isolated.on("GET", "/pods/a", body={"id": "a", "name": "tribe-a", "volumeInGb": 120})
    isolated.on("DELETE", "/pods/a", status=204, body=None)
    code, o, e = run(["pod-terminate", "a"], capsys)
    assert code == 2 and "DELETES" in e and "--yes" in e
    assert not any(c[0] == "DELETE" for c in isolated.calls)
    code, o, e = run(["pod-terminate", "a", "--yes"], capsys)
    assert code == 0 and "pod volume deleted" in o


def test_pod_terminate_network_volume_needs_no_confirmation(isolated, capsys):
    isolated.on("GET", "/pods/n", body={"id": "n", "networkVolumeId": "vol1"})
    isolated.on("DELETE", "/pods/n", status=204, body=None)
    code, o, e = run(["pod-terminate", "n"], capsys)
    assert code == 0 and "network volume kept" in o


def test_watchdog_loop_stops_pod_volume_pod(isolated, capsys, monkeypatch):
    monkeypatch.setattr(runpod, "now_ts", lambda: T0)
    runpod.write_json_atomic(runpod.STATE_FILE, {"pods": {"a": {"created_at": runpod.iso(T0 - 5 * 3600),
                                                                "max_hours": 1}}})
    isolated.on("GET", "/pods", body=[pod("a", "tribe-a", 0.34)])
    isolated.on("POST", "graphql", graphql_key="myself", body={"errors": [{"message": "down"}]})
    isolated.on("POST", "/pods/a/stop", body={"id": "a"})
    code, o, e = run(["watchdog", "--max-usd", "10", "--max-hours", "6", "--once"], capsys)
    assert code == 0 and "STOPPED a" in o and "pull" in o and "telemetry unavailable" in o
    assert not any(c[0] == "DELETE" for c in isolated.calls)
    assert json.loads(runpod.STATE_FILE.read_text())["pods"]["a"]["stopped_at"]


def test_pod_wait_falls_back_to_graphql_runtime_ports(isolated, capsys):
    isolated.on("GET", "/pods/c", body={"id": "c", "desiredStatus": "RUNNING", "publicIp": ""})
    isolated.on("POST", "graphql", graphql_key="runtime { ports", body={"data": {"myself": {"pods": [
        {"id": "other", "runtime": {"ports": [{"ip": "5.5.5.5", "isIpPublic": True, "privatePort": 22,
                                               "publicPort": 1, "type": "tcp"}]}},
        {"id": "c", "runtime": {"ports": [
            {"ip": "10.0.0.2", "isIpPublic": False, "privatePort": 22, "publicPort": 22, "type": "tcp"},
            {"ip": "203.0.113.7", "isIpPublic": True, "privatePort": 22, "publicPort": 40123, "type": "tcp"}]}}]}}})
    code, o, e = run(["pod-wait", "c"], capsys)
    assert code == 0 and o == "export POD=root@203.0.113.7 POD_PORT=40123\n"


def test_real_transport_sends_key_only_in_header(monkeypatch):
    seen = {}

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"[]"

    def fake_urlopen(req, timeout):
        seen["url"], seen["headers"], seen["timeout"] = req.full_url, dict(req.header_items()), timeout
        return Resp()

    monkeypatch.setattr(runpod.urllib.request, "urlopen", fake_urlopen)
    assert REAL_HTTP("GET", runpod.REST_URL + "/pods") == (200, b"[]")
    assert seen["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert API_KEY not in seen["url"] and seen["timeout"] == runpod.HTTP_TIMEOUT

    def boom(req, timeout):
        raise runpod.urllib.error.URLError("connection refused")

    monkeypatch.setattr(runpod.urllib.request, "urlopen", boom)
    with pytest.raises(runpod.ApiError) as ei:
        REAL_HTTP("GET", runpod.REST_URL + "/pods?x=1")
    assert "/pods" in str(ei.value) and "x=1" not in str(ei.value)
    assert_no_secrets(str(ei.value))

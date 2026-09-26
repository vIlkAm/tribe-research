"""Offline tests for tools/fleet.py. A fake ops layer simulates pods; nothing here can reach
RunPod, SSH or rsync (the autouse fixture makes every subprocess call fail)."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import fleet  # noqa: E402
import runpod  # noqa: E402

T0 = 1_790_000_000.0
REAL_RUN, REAL_POPEN = subprocess.run, subprocess.Popen  # local bash checks of generated scripts only


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError(f"subprocess call in an offline test: {a[:1]}")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(fleet, "git_rev", lambda: "test")
    monkeypatch.setattr(runpod, "HEARTBEAT_FILE", tmp_path / "no-heartbeat.json")
    monkeypatch.setattr(runpod, "STATE_FILE", tmp_path / "no-runpod-state.json")
    monkeypatch.setattr(runpod, "load_env", lambda path=None: {})


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t


def make_batch(root: Path, name: str, n: int, ready: bool = True, missing: int = 0) -> Path:
    d = root / "batches" / name
    (d / "videos" / "deal").mkdir(parents=True)
    rows = []
    for i in range(n):
        vid = f"{name}v{i:03d}"
        rel = f"deal/{vid}.mp4"
        if i >= missing:
            (d / "videos" / rel).write_bytes(b"x")
        rows.append({"video_id": vid, "path": rel, "source_name": vid, "duration_s": 10.0 + i,
                     "size_bytes": 1, "worker": i % 2, "num_workers": 2})
    (d / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if ready:
        (d / "prep.json").write_text("{}")
    return d


class FakeOps:
    """Pods as dicts. status() advances the simulation: setup counts down, each alive worker
    finishes `per_poll` clips per call. Every call is appended to `events`."""

    def __init__(self, clock, setup_polls=3, per_poll=2, rate=1.0):
        self.clock = clock
        self.setup_polls = setup_polls
        self.per_poll = per_poll
        self.rate = rate
        self.pods: dict[str, dict] = {}
        self.events: list[tuple] = []
        self.unreachable: set[str] = set()
        self.api_status: dict[str, str] = {}

    def ev(self, *e):
        self.events.append(e)

    def idx(self, *e):
        return self.events.index(e)

    # API
    def watchdog(self, now):
        return True, "watchdog alive", {"max_usd": 1000, "max_hours": 100}

    def create_pod(self, cfg, name):
        pid = f"pod{len(self.pods)}"
        self.pods[pid] = {"name": name, "setup": None, "batches": {}, "terminated": False}
        self.ev("create", pid)
        return pid, self.rate

    def list_pods(self):
        return {pid: {"name": p["name"], "status": self.api_status.get(pid, "RUNNING"), "rate": self.rate,
                      "terminated": p["terminated"]} for pid, p in self.pods.items()}

    def recorded_rate(self, pod_id):
        return None

    def wait_ssh(self, pod_id, timeout_min):
        return {"pod_id": pod_id, "host": f"root@{pod_id}", "port": 22}

    def terminate(self, pod_id):
        self.pods[pod_id]["terminated"] = True
        self.ev("terminate", pod_id)

    def stop(self, pod_id):
        self.ev("stop", pod_id)

    # SSH
    def _reach(self, ep):
        if ep["pod_id"] in self.unreachable:
            raise fleet.SshError("connection refused")
        return self.pods[ep["pod_id"]]

    def push_code(self, ep, job):
        self._reach(ep)
        self.ev("push_code", ep["pod_id"])

    def start_setup(self, ep, job):
        self._reach(ep)["setup"] = self.setup_polls
        self.ev("setup_start", ep["pod_id"])
        return "FLEET setup started"

    def push_batch(self, ep, job, name, videos_dir, files, manifest):
        pod = self._reach(ep)
        rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
        assert Path(files).read_text().split() == [r["path"] for r in rows]
        pod["batches"][name] = {"rows": rows, "alive": set(), "done": set()}
        self.ev("push", ep["pod_id"], name)

    def launch(self, ep, job, batch, workers, cfg):
        b = self._reach(ep)["batches"][batch]
        b["alive"] |= set(workers)
        self.ev("launch", ep["pod_id"], batch)
        return "FLEET launched"

    def status(self, ep, job, names, ids_for=()):
        pod = self._reach(ep)
        if pod["setup"] is not None and pod["setup"] > 0:
            pod["setup"] -= 1
            if pod["setup"] == 0:
                self.ev("setup_complete", ep["pod_id"])
        out = {"setup": "complete" if pod["setup"] == 0 else "running", "free_gb": 80.0, "batches": {}}
        for name in names:
            b = pod["batches"].get(name)
            if b is None:
                out["batches"][name] = {"manifest": False, "workers": {}}
                continue
            for k in sorted(b["alive"]):
                todo = [r["video_id"] for r in b["rows"] if r["worker"] == k and r["video_id"] not in b["done"]]
                b["done"] |= set(todo[: self.per_poll])
                if len(todo) <= self.per_poll:
                    b["alive"].discard(k)
            if b["rows"] and not b["alive"] and len(b["done"]) == len(b["rows"]) and ("all_done", ep["pod_id"], name) not in self.events:
                self.ev("all_done", ep["pod_id"], name)
            workers = {}
            for r in b["rows"]:
                w = workers.setdefault(str(r["worker"]), {"shard": 0, "done": 0, "errors": 0, "alive": False})
                w["shard"] += 1
                w["done"] += r["video_id"] in b["done"]
                w["alive"] = r["worker"] in b["alive"]
            info = {"manifest": True, "outputs": bool(b["done"]), "total": len(b["rows"]), "done": len(b["done"]),
                    "errors": 0, "workers": workers}
            if name in ids_for:
                info["done_ids"] = sorted(b["done"])
            out["batches"][name] = info
        return out

    def pull_batch(self, ep, job, name, dest):
        b = self._reach(ep)["batches"][name]
        for r in b["rows"]:
            if r["video_id"] in b["done"]:
                d = Path(dest) / f"worker-{r['worker']}"
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{r['video_id']}.npz").write_bytes(b"p")
                (d / f"{r['video_id']}.json").write_text("{}")
        self.ev("pull", ep["pod_id"], name)

    def pull_logs(self, ep, job, dest, setup_only=False):
        self.ev("pull_logs", ep["pod_id"])

    def cleanup_batch(self, ep, job, name):
        self.ev("cleanup", ep["pod_id"], name)


def cfg(**kw):
    base = dict(pods=1, gpu_type="L40S", workers_per_gpu=2, max_usd=100.0, usd_per_hour=1.0, poll_s=60,
                pull_every_min=10**6, idle_grace_min=10, dead_after_min=10, create_backoff_s=0)
    base.update(kw)
    return fleet.Config(**base)


def make_fleet(tmp_path, clock, ops, **kw):
    f = fleet.Fleet(cfg(**kw), ops, tmp_path / "fleet", now=clock, echo=False)
    return f


def drive(f, clock, rounds=400, dt=60):
    f.ensure_slots()
    for _ in range(rounds):
        active = [s for s in sorted(f.state["slots"]) if f.state["slots"][s]["phase"] != "ended"]
        if not active:
            return
        f.api_tick()
        for s in active:
            f.step(s)
        clock.t += dt
    raise AssertionError("fleet did not finish")


# --------------------------------------------------------------------------- queue

def test_queue_claim_release_resume(tmp_path):
    clock = Clock()
    b1, b2 = make_batch(tmp_path, "b1", 4), make_batch(tmp_path, "b2", 4)
    make_batch(tmp_path, "b3", 4, ready=False)
    f = make_fleet(tmp_path, clock, FakeOps(clock))
    f.add_batches([b1, b2, tmp_path / "batches" / "b3"])
    f.ensure_slots()
    assert [f.state["batches"][n]["status"] for n in ("b1", "b2", "b3")] == ["queued", "queued", "waiting_prep"]
    name, rows, _ = f.claim("s0")
    assert name == "b1" and len(rows) == 4

    # driver restart: a new Fleet reads the same state file
    g = make_fleet(tmp_path, clock, FakeOps(clock))
    assert g.state["batches"]["b1"]["status"] == "claimed" and g.state["batches"]["b1"]["slot"] == "s0"
    assert g.claim("s0")[0] == "b2"  # b1 is taken
    g.release("b1", "test")
    assert g.state["batches"]["b1"]["status"] == "queued" and g.state["batches"]["b1"]["attempts"] == 1
    assert g.claim("s0")[0] == "b1"
    # prep finishes later: the batch becomes claimable
    (tmp_path / "batches" / "b3" / "prep.json").write_text("{}")
    assert g.claim("s0")[0] == "b3"
    assert g.claim("s0") is None


def test_release_gives_up_after_max_attempts(tmp_path):
    clock = Clock()
    f = make_fleet(tmp_path, clock, FakeOps(clock), max_batch_attempts=2)
    f.add_batches([make_batch(tmp_path, "b1", 2)])
    f.ensure_slots()
    for _ in range(2):
        assert f.claim("s0")[0] == "b1"
        f.release("b1", "pod died")
    assert f.state["batches"]["b1"]["status"] == "failed"
    assert f.claim("s0") is None


# --------------------------------------------------------------------------- manifest

def test_pod_manifest_reassigns_workers_and_drops_done_and_missing(tmp_path):
    b = make_batch(tmp_path, "b1", 12, missing=1)  # row 0 has no local video
    rows, info = fleet.build_pod_manifest(b, {"b1v001", "b1v002"}, num_workers=4)
    assert info == {"total": 12, "done": 2, "missing": ["deal/b1v000.mp4"]}
    assert len(rows) == 9
    assert {r["num_workers"] for r in rows} == {4}
    assert {r["worker"] for r in rows} == {0, 1, 2, 3}
    loads = [sum(r["duration_s"] for r in rows if r["worker"] == w) for w in range(4)]
    assert max(loads) - min(loads) <= max(r["duration_s"] for r in rows)
    # the source manifest is untouched
    assert json.loads((b / "manifest.jsonl").read_text().splitlines()[0])["num_workers"] == 2


def test_local_done_needs_json_and_preds(tmp_path):
    d = tmp_path / "out" / "worker-0"
    d.mkdir(parents=True)
    (d / "a.json").write_text("{}")
    (d / "a.npz").write_bytes(b"p")
    (d / "b.json").write_text("{}")          # preds missing: not done
    (d / "c.error.json").write_text("{}")
    assert fleet.local_done_ids(tmp_path / "out") == {"a"}


# --------------------------------------------------------------------------- lifecycle

def test_lifecycle_overlaps_setup_and_prefetches(tmp_path):
    clock = Clock()
    ops = FakeOps(clock, setup_polls=3, per_poll=1)
    f = make_fleet(tmp_path, clock, ops)
    f.add_batches([make_batch(tmp_path, "b1", 6), make_batch(tmp_path, "b2", 6), make_batch(tmp_path, "b3", 2)])
    drive(f, clock)

    # code (and rsync) before setup, so setup.sh's apt-get never races ours
    assert ops.idx("push_code", "pod0") < ops.idx("setup_start", "pod0")
    # first batch pushed while setup.sh runs
    assert ops.idx("setup_start", "pod0") < ops.idx("push", "pod0", "b1") < ops.idx("setup_complete", "pod0")
    # workers start only after setup
    assert ops.idx("setup_complete", "pod0") < ops.idx("launch", "pod0", "b1")
    # b2 prefetched while b1 runs
    assert ops.idx("launch", "pod0", "b1") < ops.idx("push", "pod0", "b2") < ops.idx("all_done", "pod0", "b1")
    # b2 launched before b1 is pulled: the GPU never waits on the network
    assert ops.idx("all_done", "pod0", "b1") < ops.idx("launch", "pod0", "b2") < ops.idx("pull", "pod0", "b1")
    assert ops.idx("launch", "pod0", "b3") < ops.idx("pull", "pod0", "b2")
    # teardown only after every pull; one pod, terminated once
    assert ops.idx("pull", "pod0", "b3") < ops.idx("terminate", "pod0")
    assert [e for e in ops.events if e[0] in ("create", "terminate")] == [("create", "pod0"), ("terminate", "pod0")]

    st = f.state
    assert all(st["batches"][n]["status"] == "done" for n in ("b1", "b2", "b3"))
    assert fleet.local_done_ids(tmp_path / "fleet" / "outputs" / "b1") == {f"b1v{i:03d}" for i in range(6)}
    assert st["pods"]["pod0"]["ended_at"] is not None
    assert "clips/h" in fleet.status_text(fleet.Store(tmp_path / "fleet" / "state.json").load(), clock.t)
    log = (tmp_path / "fleet" / "fleet.log").read_text()
    for ev in ("pod_created", "setup_done", "push_done", "launch", "pull_verified", "batch_done", "pod_terminate"):
        assert f" {ev} " in log


def test_multiple_pods_share_the_queue(tmp_path):
    clock = Clock()
    ops = FakeOps(clock, setup_polls=2, per_poll=2)
    f = make_fleet(tmp_path, clock, ops, pods=2)
    f.add_batches([make_batch(tmp_path, f"b{i}", 4) for i in range(5)])
    drive(f, clock)
    pushed = [e[2] for e in ops.events if e[0] == "push"]
    assert sorted(pushed) == [f"b{i}" for i in range(5)]  # each batch pushed exactly once
    assert {e[1] for e in ops.events if e[0] == "push"} == {"pod0", "pod1"}
    assert all(b["status"] == "done" for b in f.state["batches"].values())
    assert sorted(e[1] for e in ops.events if e[0] == "terminate") == ["pod0", "pod1"]


def test_dead_pod_requeues_batch_and_skips_pulled_clips(tmp_path):
    clock = Clock()
    ops = FakeOps(clock, setup_polls=1, per_poll=1)
    f = make_fleet(tmp_path, clock, ops, pull_every_min=1, dead_after_min=5)
    f.add_batches([make_batch(tmp_path, "b1", 8)])
    f.ensure_slots()
    for _ in range(40):  # run until a partial pull brought some clips home
        f.step("s0")
        clock.t += 60
        if fleet.local_done_ids(tmp_path / "fleet" / "outputs" / "b1") and f.state["slots"]["s0"]["phase"] == "run":
            break
    pulled = fleet.local_done_ids(tmp_path / "fleet" / "outputs" / "b1")
    assert 0 < len(pulled) < 8
    ops.unreachable.add("pod0")
    for _ in range(8):
        f.step("s0")
        clock.t += 60
        if f.state["batches"]["b1"]["status"] != "claimed" or f.state["slots"]["s0"]["phase"] == "wait_ssh":
            break
    assert ("terminate", "pod0") in ops.events
    b = f.state["batches"]["b1"]
    assert b["attempts"] == 1
    # the slot replaced the pod; the replacement got only the clips not already pulled
    drive(f, clock)
    assert ("create", "pod1") in ops.events
    assert len(ops.pods["pod1"]["batches"]["b1"]["rows"]) == 8 - len(pulled)
    assert f.state["batches"]["b1"]["status"] == "done"
    assert fleet.local_done_ids(tmp_path / "fleet" / "outputs" / "b1") == {f"b1v{i:03d}" for i in range(8)}


def test_pod_stopped_by_watchdog_is_detected_via_api(tmp_path):
    clock = Clock()
    ops = FakeOps(clock, setup_polls=1, per_poll=1)
    f = make_fleet(tmp_path, clock, ops, max_creates=1)
    f.add_batches([make_batch(tmp_path, "b1", 20)])
    f.ensure_slots()
    for _ in range(8):
        f.api_tick()
        f.step("s0")
        clock.t += 60
    assert f.state["batches"]["b1"]["status"] == "claimed"
    ops.api_status["pod0"] = "EXITED"
    clock.t += 600
    f.api_tick()
    f.step("s0")
    assert f.state["batches"]["b1"]["status"] == "queued"
    assert ("terminate", "pod0") in ops.events  # a stopped pod's container disk is gone anyway
    f.step("s0")
    assert f.state["slots"]["s0"]["phase"] == "ended"  # max-creates 1: no replacement


def test_setup_failure_and_timeout_terminate_after_log_pull(tmp_path):
    clock = Clock()
    ops = FakeOps(clock, setup_polls=10**6)
    f = make_fleet(tmp_path, clock, ops, setup_timeout_min=30, max_creates=1)
    f.add_batches([make_batch(tmp_path, "b1", 4)])
    drive(f, clock, dt=300)
    assert ops.idx("pull_logs", "pod0") < ops.idx("terminate", "pod0")
    assert f.state["batches"]["b1"]["status"] == "queued"
    assert "setup timeout" in (tmp_path / "fleet" / "fleet.log").read_text()


# --------------------------------------------------------------------------- money / ownership

def test_max_usd_hard_cap_pulls_and_terminates(tmp_path):
    clock = Clock()
    ops = FakeOps(clock, setup_polls=1, per_poll=1, rate=2.0)
    f = make_fleet(tmp_path, clock, ops, max_usd=3.0, reserve_hours=0.0)
    f.add_batches([make_batch(tmp_path, "b1", 400), make_batch(tmp_path, "b2", 4)])
    drive(f, clock, rounds=400, dt=600)
    assert [e[1] for e in ops.events if e[0] == "create"] == ["pod0"]  # no new pod at the cap
    assert ("terminate", "pod0") in ops.events
    assert ops.idx("pull", "pod0", "b1") < ops.idx("terminate", "pod0")  # partial outputs saved first
    assert f.state["batches"]["b1"]["status"] == "queued"
    assert f.budget() == "hard"
    usd, _ = f.spend()
    assert usd < 3.0 + 2.0 * 600 / 3600 + 1e-9  # overshoot at most one poll interval of burn


def test_budget_reserve_blocks_pod_creation(tmp_path):
    clock = Clock()
    ops = FakeOps(clock)
    f = make_fleet(tmp_path, clock, ops, max_usd=0.2, usd_per_hour=1.0, reserve_hours=0.25)
    f.add_batches([make_batch(tmp_path, "b1", 4)])
    drive(f, clock)
    assert not [e for e in ops.events if e[0] == "create"]
    assert "budget soft" in (tmp_path / "fleet" / "fleet.log").read_text()


def test_idle_pod_is_terminated_when_budget_reserve_reached(tmp_path):
    clock = Clock()
    ops = FakeOps(clock, setup_polls=1, per_poll=10)
    # b2 is still being prepped, so the pod would wait for it (idle grace is huge) ...
    f = make_fleet(tmp_path, clock, ops, max_usd=1.0, reserve_hours=0.25, idle_grace_min=10**6)
    f.add_batches([make_batch(tmp_path, "b1", 4), make_batch(tmp_path, "b2", 4, ready=False)])
    drive(f, clock, dt=60)
    # ... until the budget reserve is reached: then the idle pod is terminated, no new pod
    assert f.state["batches"]["b1"]["status"] == "done"
    assert ops.idx("pull", "pod0", "b1") < ops.idx("terminate", "pod0")
    assert [e for e in ops.events if e[0] == "create"] == [("create", "pod0")]
    assert "budget reserve reached" in (tmp_path / "fleet" / "fleet.log").read_text()
    assert f.state["slots"]["s0"]["phase"] == "ended"


def test_never_ends_an_unowned_pod(tmp_path):
    clock = Clock()
    ops = FakeOps(clock)
    f = make_fleet(tmp_path, clock, ops)
    f.ensure_slots()
    f.state["slots"]["s0"].update(pod_id="someone-elses-pod", ep={"pod_id": "x", "host": "h", "port": 1})
    f.end_pod("s0", "terminate", "test")
    assert not [e for e in ops.events if e[0] in ("terminate", "stop")]
    assert "refuse_end_unowned_pod" in (tmp_path / "fleet" / "fleet.log").read_text()


def test_create_failure_adopts_exact_name_instead_of_retrying(tmp_path):
    clock = Clock()

    class LostCreate(FakeOps):
        def create_pod(self, cfg, name):
            super().create_pod(cfg, name)  # the pod exists, but the reply was lost
            raise fleet.ApiFailure("timeout")

    ops = LostCreate(clock)
    f = make_fleet(tmp_path, clock, ops)
    f.add_batches([make_batch(tmp_path, "b1", 2)])
    f.ensure_slots()
    f.step("s0")
    assert [e for e in ops.events if e[0] == "create"] == [("create", "pod0")]
    assert f.state["slots"]["s0"]["pod_id"] == "pod0" and "pod0" in f.state["pods"]


# --------------------------------------------------------------------------- plan / preflight

class RaisingOps:
    def __getattr__(self, name):
        raise AssertionError(f"ops.{name} called during a dry run")


def test_dry_run_prints_plan_without_any_call(tmp_path, capsys):
    make_batch(tmp_path, "r00", 10)
    make_batch(tmp_path, "r01", 6)
    make_batch(tmp_path, "r02", 6, ready=False)
    make_batch(tmp_path, "r03", 6)
    # defaults: bf16 fast-video, workers from VRAM, rate from the measured 1.1 s per source-second
    rc = fleet.main(["run", "--dry-run", "--pods", "2", "--gpu-type", "L40S", "--max-usd", "50",
                     "--usd-per-hour", "0.9", "--exclude", "r03",
                     "--batches", str(tmp_path / "batches" / "r*")], ops=RaisingOps(), fleet_dir=tmp_path / "fleet")
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 ready, 1 waiting for prep (r02), 1 skipped (r03)" in out
    assert "ready work: 16 clips" in out  # 220 source-seconds, mean 14 s
    # per worker 3600 / (1.1 * 13.75) = 238/h; 2 workers per L40S; 2 pods
    assert "1.1 s wall per source-second per worker" in out and "238 clips/h per worker, 476/pod, 952/fleet" in out
    assert "2/GPU (auto: (48 GB - 3) / ~19 GB) = 2/pod" in out
    assert "max(4, cgroup CPU quota / 2)" in out and "expandable_segments:True" in out
    assert "cost: ~$" in out and "JOB=/root/tribe-job" in out
    assert "flags: --fast-video --video-precision bf16" in out and "feature-cache-bf16" in out
    assert "--container-disk-gb 100" in out
    assert not (tmp_path / "fleet").exists()  # no state, no log


def test_run_without_go_is_a_plan(tmp_path, capsys):
    rc = fleet.main(["run", "--gpu-type", "L40S", "--workers-per-gpu", "1"], ops=RaisingOps(),
                    fleet_dir=tmp_path / "fleet")
    assert rc == 0 and "add --go" in capsys.readouterr().out


def test_config_refuses_two_workers_on_24gb_and_bf16_without_fast_video():
    problems = fleet.Config(gpu_type="4090", workers_per_gpu=2).validate()
    assert any("do not fit" in p for p in problems)
    assert fleet.Config(gpu_type="4090", workers_per_gpu=1).validate() == []
    assert any("--fast-video" in p for p in fleet.Config(gpu_type="L40S", fast_video=False).validate())
    # measured defaults: bf16 fast-video; ~19 GB per worker decides workers per GPU
    c = fleet.Config(gpu_type="L40S")
    assert (c.fast_video, c.video_precision, c.wpg, c.num_workers) == (True, "bf16", 2, 2)
    assert c.worker_flags() == ["--fast-video", "--video-precision", "bf16"] and c.cache_folder.endswith("-bf16")
    assert fleet.Config(gpu_type="4090").wpg == 1 and fleet.Config(gpu_type="4090").validate() == []
    assert fleet.Config(gpu_type="A6000").wpg == 2 and fleet.Config(gpu_type="A100 80GB PCIe").wpg == 4
    assert fleet.Config(gpu_type="L40S", gpu_count=2).num_workers == 4
    assert fleet.Config(gpu_type="L40S", workers_per_gpu=1).wpg == 1  # override
    assert any("Blackwell" in p for p in fleet.Config(gpu_type="RTX 5090").validate())


def test_preflight_requires_watchdog_with_wider_caps():
    class Ops:
        def __init__(self, ok, hb):
            self.r = (ok, "why", hb)

        def watchdog(self, now):
            return self.r

    c = cfg(max_usd=50, pod_max_hours=8)
    assert any("watchdog" in p for p in fleet.preflight(c, Ops(False, {}), T0))
    low = fleet.preflight(c, Ops(True, {"max_usd": 7.5, "max_hours": 3}), T0)
    assert any("--max-usd 7.5" in p for p in low) and any("--max-hours 3" in p for p in low)
    assert fleet.preflight(c, Ops(True, {"max_usd": 60, "max_hours": 10}), T0) == []


# --------------------------------------------------------------------------- remote side

def test_every_remote_script_exports_job():
    job = "/root/tribe-job"
    scripts = [fleet.setup_script(job),
               fleet.launch_script(job, "r00", [0, 1, 2, 3], 4, 2, f"{job}/feature-cache-bf16",
                                   ["--fast-video", "--video-precision", "bf16"])]
    for s in scripts:
        assert s.startswith("set -eu\nexport JOB=/root/tribe-job\n")
    setup, launch = scripts
    assert "setsid nohup" in setup and "< /dev/null &" in setup and "logs/setup.out" in setup
    assert ".setup_done" in setup
    lines = [l for l in launch.splitlines() if "run_worker.sh" in l]
    assert len(lines) == 4
    for k, line in enumerate(lines):
        assert f"WORKER_ID={k} NUM_WORKERS=4 GPU={k % 2} " in line
        assert line.endswith("< /dev/null &")
        assert "--manifest /root/tribe-job/fleet/batches/r00.jsonl" in line
        assert "--out-root /root/tribe-job/outputs/r00" in line
        assert "--cache-folder /root/tribe-job/feature-cache-bf16 --fast-video --video-precision bf16" in line


def test_remote_status_script_reads_a_job_tree(tmp_path):
    job = tmp_path / "job"
    (job / "fleet" / "batches").mkdir(parents=True)
    rows = [{"video_id": f"v{i}", "path": f"d/v{i}.mp4", "duration_s": 5, "worker": i % 2, "num_workers": 2}
            for i in range(4)]
    (job / "fleet" / "batches" / "r00.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    w0 = job / "outputs" / "r00" / "worker-0"
    w0.mkdir(parents=True)
    for name in ("v0.json", "v0.npz", "v0.emb.npz", "v2.error.json"):
        (w0 / name).write_text("{}")
    (job / "fleet" / "setup.rc").write_text("1\n")
    (job / ".setup_failed").write_text("rc=1 step=spacy model\n")
    arg = json.dumps({"job": str(job), "batches": ["r00", "r99"], "ids_for": ["r00"]})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(fleet.REMOTE_STATUS % repr(arg), "remote_status", "exec"), {})
    line = buf.getvalue().strip()
    assert line.startswith("FLEETSTATUS ")
    st = json.loads(line[len("FLEETSTATUS "):])
    assert st["setup"] == "failed" and "spacy" in st["setup_failed"]
    b = st["batches"]["r00"]
    assert (b["total"], b["done"], b["errors"], b["emb"]) == (4, 1, 1, 1)
    assert b["done_ids"] == ["v0"]
    assert b["workers"]["0"] == {"shard": 2, "done": 1, "errors": 1, "alive": False}
    assert st["batches"]["r99"]["manifest"] is False
    (job / ".setup_done").write_text("")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(fleet.REMOTE_STATUS % repr(arg), "remote_status", "exec"), {})
    assert json.loads(buf.getvalue().split(" ", 1)[1])["setup"] == "complete"


def test_real_ops_exports_job_and_uses_private_known_hosts(tmp_path, monkeypatch):
    calls = []

    def runner(argv, input=None, timeout=None, env=None):
        calls.append((argv, input, env))
        return 0, "FLEETSTATUS " + json.dumps({"setup": "running", "batches": {}}), ""

    ops = fleet.RealOps(tmp_path / "fleet", "/root/tribe-job", runner=runner)
    ep = {"pod_id": "abc", "host": "root@1.2.3.4", "port": 4242}
    ops.status(ep, "/root/tribe-job", [])
    ops.pull_batch(ep, "/root/tribe-job", "r00", tmp_path / "out")
    ssh_argv, script, env = calls[0]
    assert env == {"JOB": "/root/tribe-job"}
    assert ssh_argv[0] == "ssh" and "BatchMode=yes" in ssh_argv and "-p" in ssh_argv
    assert f"UserKnownHostsFile={tmp_path / 'fleet' / 'known_hosts' / 'abc'}" in ssh_argv
    assert '"job": "/root/tribe-job"' in script
    rsync_argv = calls[1][0]
    assert rsync_argv[:2] == ["rsync", "-rlpt"] and calls[1][2] == {"JOB": "/root/tribe-job"}
    assert rsync_argv[-2:] == ["root@1.2.3.4:/root/tribe-job/outputs/r00/", f"{tmp_path / 'out'}/"]


def test_real_ops_ssh_failure_is_connection_level(tmp_path):
    ops = fleet.RealOps(tmp_path, "/root/tribe-job", runner=lambda *a, **k: (255, "", "Connection refused"))
    with pytest.raises(fleet.SshError):
        ops.status({"pod_id": "p", "host": "root@h", "port": 1}, "/root/tribe-job", [])
    ops = fleet.RealOps(tmp_path, "/root/tribe-job", runner=lambda *a, **k: (1, "", "No such file"))
    with pytest.raises(fleet.RemoteError):
        ops.cleanup_batch({"pod_id": "p", "host": "root@h", "port": 1}, "/root/tribe-job", "r00")


# --------------------------------------------------------------------------- measured pod defaults

@pytest.fixture
def local_bash(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", REAL_POPEN)


def _bash(script: str, env: dict) -> str:
    p = REAL_RUN(["bash", "-c", script], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env},
                 timeout=20)
    assert p.returncode == 0, p.stderr
    return p.stdout.strip()


def test_threads_from_cgroup_quota_not_nproc(tmp_path, local_bash):
    v2 = tmp_path / "v2"
    v2.mkdir()
    (v2 / "cpu.max").write_text("2720000 100000\n")  # 27.2 CPUs, as measured on the L40S pod
    run = lambda w, root, override=None: _bash(  # noqa: E731
        "set -eu\n" + fleet.threads_script(w, override) + 'echo "$CPUS $T"', {"FLEET_CGROUP_ROOT": str(root)})
    assert run(2, v2) == "27 13"
    assert run(8, v2) == "27 4"  # floor 3 -> at least 4
    v1 = tmp_path / "v1"
    (v1 / "cpu").mkdir(parents=True)
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("1600000\n")
    (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert run(2, v1) == "16 8"
    (v1 / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")  # unlimited: nproc fallback
    nproc = int(_bash("nproc", {}))
    assert run(2, v1) == f"{nproc} {max(4, nproc // 2)}"
    (v2 / "cpu.max").write_text("max 100000\n")
    assert run(1, v2) == f"{nproc} {max(4, nproc)}"
    assert run(2, v2, override=6) == "override 6"


def test_launch_script_sets_threads_and_allocator(tmp_path, local_bash):
    job = "/root/tribe-job"
    s = fleet.launch_script(job, "r00", [0, 1], 2, 1, f"{job}/feature-cache-bf16", fleet.Config().worker_flags())
    head = s.split("WORKER_ID=0")[0]
    assert 'export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T"' in head
    assert 'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"' in head
    assert "TRIBE_VIDEO_THREADS" in head and "W=2" in head
    assert "--fast-video --video-precision bf16" in s
    assert "cpus $CPUS threads $T" in s
    REAL_RUN(["bash", "-n"], input=s, text=True, check=True)  # syntax only, nothing executed
    # evaluate the env block (everything before the first worker launch) against a fake quota
    (tmp_path / "cpu.max").write_text("2720000 100000\n")
    probe = 'echo "$OMP_NUM_THREADS $MKL_NUM_THREADS $TRIBE_VIDEO_THREADS $PYTORCH_CUDA_ALLOC_CONF"'
    for w, want in ((2, "13 13 8 expandable_segments:True"), (8, "4 4 4 expandable_segments:True")):
        env_block = fleet.launch_script(job, "r00", [0], w, 1, "c", []).split("WORKER_ID=")[0]
        assert _bash(env_block.replace("mkdir -p", ": mkdir -p") + probe, {"FLEET_CGROUP_ROOT": str(tmp_path)}) == want


# --------------------------------------------------------------------------- queue order / exclude / add

def test_default_queue_order():
    names = [p.name for p in fleet.default_batches()]
    assert names[:10] == ["b02", "b03", "b04", "b05", "b06", "b07", "b08", "b09", "d00_deep_dive", "d01_deep_dive"]
    assert names[10:] == [f"r{i:02d}" for i in range(16)]
    assert all(p.parent == fleet.ROOT / "results" / "batches_s384" for p in fleet.default_batches())


def test_exclude_and_runtime_add(tmp_path, capsys):
    clock = Clock()
    f = make_fleet(tmp_path, clock, FakeOps(clock))
    b = [make_batch(tmp_path, n, 2) for n in ("b02", "b03", "r00")]
    f.add_batches(b)
    f.ensure_slots()
    f.exclude(["b02", "b03"])  # e.g. run by a manually driven pod
    assert [f.state["batches"][n]["status"] for n in ("b02", "b03", "r00")] == ["excluded", "excluded", "queued"]
    assert f.claim("s0")[0] == "r00"
    assert f.claim("s0") is None
    f.add_batches([b[0]])  # a later add/rescan does not bring an excluded batch back
    assert f.state["batches"]["b02"]["status"] == "excluded"

    # runtime requests from another process go through the inbox
    fd = tmp_path / "fleet"
    assert fleet.main(["add", "b03", str(tmp_path / "batches" / "n*")], fleet_dir=fd) == 0
    assert fleet.main(["exclude", "r00,b02"], fleet_dir=fd) == 0
    assert "inbox (merged at the driver's next tick): add b03" in fleet.status_text(
        fleet.Store(fd / "state.json").load(), clock.t, inbox=fleet.Inbox(fd).peek())
    make_batch(tmp_path, "n01", 2)
    f.merge_inbox()
    assert not (fd / "inbox.jsonl").exists()
    assert f.state["batches"]["b03"]["status"] == "queued"  # exclusion undone
    assert f.state["batches"]["n01"]["status"] == "queued"
    assert f.state["batches"]["r00"]["status"] == "claimed"  # running: deferred, finishes
    assert "batch_exclude_deferred" in (fd / "fleet.log").read_text()
    f.release("r00", "pod died")
    assert f.state["batches"]["r00"]["status"] == "excluded"  # ... but is not requeued
    # the glob is rescanned while running: a batch dir that appears later joins the queue
    make_batch(tmp_path, "n02", 2, ready=False)
    f.rescan()
    assert f.state["order"][-1] == "n02" and f.state["batches"]["n02"]["status"] == "waiting_prep"
    assert [f.claim("s0")[0] for _ in range(2)] == ["b03", "n01"]


def test_ended_slot_revives_when_work_is_added(tmp_path):
    clock = Clock()
    ops = FakeOps(clock, setup_polls=1, per_poll=4)
    f = make_fleet(tmp_path, clock, ops)
    f.add_batches([make_batch(tmp_path, "b1", 2)])
    drive(f, clock)
    assert f.state["slots"]["s0"]["phase"] == "ended" and f.revive_slots() == []
    fleet.Inbox(tmp_path / "fleet").put("add", [str(make_batch(tmp_path, "b2", 2))])
    f.merge_inbox()
    assert f.revive_slots() == ["s0"]
    drive(f, clock)
    assert f.state["batches"]["b2"]["status"] == "done"
    assert [e[1] for e in ops.events if e[0] == "create"] == ["pod0", "pod1"]


# --------------------------------------------------------------------------- failure paths

class ZeroOutputOps(FakeOps):
    """Workers start and exit without producing anything (broken env / OOM at load)."""

    def status(self, ep, job, names, ids_for=()):
        for n in names:
            b = self._reach(ep)["batches"].get(n)
            if b:
                b["alive"].clear()
        return super().status(ep, job, names, ids_for)

    def pull_batch(self, ep, job, name, dest):
        raise fleet.RemoteError("rsync: change_dir outputs/%s failed: No such file or directory (23)" % name)


def test_zero_output_batch_halts_fleet_and_requeues(tmp_path):
    clock = Clock()
    ops = ZeroOutputOps(clock, setup_polls=1)
    f = make_fleet(tmp_path, clock, ops, pods=1, max_creates=4)
    f.add_batches([make_batch(tmp_path, "b1", 6), make_batch(tmp_path, "b2", 6)])
    drive(f, clock)
    assert f.state["halt"] and "0 of 6" in f.state["halt"]["reason"]
    assert ops.idx("pull_logs", "pod0") < ops.idx("terminate", "pod0")
    assert [e for e in ops.events if e[0] == "create"] == [("create", "pod0")]  # halted: no replacement
    assert f.state["batches"]["b1"]["status"] == "queued" and f.state["batches"]["b1"]["attempts"] == 1
    assert f.state["batches"]["b2"]["status"] == "queued"  # the prefetched batch is released too
    assert f.state["slots"]["s0"]["phase"] == "ended"
    assert "HALTED" in fleet.status_text(f.state, clock.t)


def test_tiny_batch_without_outputs_is_not_stuck_in_pending_pulls(tmp_path):
    clock = Clock()
    ops = ZeroOutputOps(clock, setup_polls=1)
    f = make_fleet(tmp_path, clock, ops, max_batch_attempts=2)
    f.add_batches([make_batch(tmp_path, "b1", 2)])  # below ZERO_OUTPUT_MIN_CLIPS: no halt
    drive(f, clock)
    assert not [e for e in ops.events if e[0] == "pull"]  # outputs dir absent: no rsync (rc 23)
    assert f.state["batches"]["b1"]["status"] == "failed" and not f.state.get("halt")
    assert ("terminate", "pod0") in ops.events and f.state["slots"]["s0"]["pending_pulls"] == []


def test_setup_failed_marker_halts_but_timeout_allows_replacement(tmp_path):
    clock = Clock()

    class SetupFails(FakeOps):
        def status(self, ep, job, names, ids_for=()):
            st = super().status(ep, job, names, ids_for)
            return {**st, "setup": "failed", "setup_failed": "rc=1 step=venv"}

    ops = SetupFails(clock, setup_polls=10**6)
    f = make_fleet(tmp_path, clock, ops, max_creates=3)
    f.add_batches([make_batch(tmp_path, "b1", 4)])
    drive(f, clock)
    assert ops.idx("pull_logs", "pod0") < ops.idx("terminate", "pod0")
    assert "step=venv" in f.state["halt"]["reason"]
    assert [e for e in ops.events if e[0] == "create"] == [("create", "pod0")]
    assert f.state["batches"]["b1"]["status"] == "queued"

    clock2 = Clock()
    ops2 = FakeOps(clock2, setup_polls=10**6)  # slow, not failed: the timeout path replaces the pod
    g = fleet.Fleet(cfg(max_creates=2, setup_timeout_min=30), ops2, tmp_path / "fleet2", now=clock2, echo=False)
    g.add_batches([make_batch(tmp_path / "x", "b1", 4)])
    drive(g, clock2, dt=300)
    assert not g.state.get("halt")
    assert [e[1] for e in ops2.events if e[0] == "create"] == ["pod0", "pod1"]


def test_failed_terminate_stays_billed_and_is_retried(tmp_path):
    clock = Clock()

    class FlakyTerminate(FakeOps):
        fails = 10**6

        def terminate(self, pod_id):
            if self.fails:
                self.fails -= 1
                self.ev("terminate_failed", pod_id)
                raise fleet.ApiFailure("runpod.py pod-terminate rc=1: 502")
            super().terminate(pod_id)

    ops = FlakyTerminate(clock, setup_polls=1, per_poll=4)
    f = make_fleet(tmp_path, clock, ops)
    f.add_batches([make_batch(tmp_path, "b1", 2)])
    drive(f, clock)  # every retry from api_tick fails too
    rec = f.state["pods"]["pod0"]
    assert "pod_terminate_retry_failed" in (tmp_path / "fleet" / "fleet.log").read_text()
    assert rec["ended_at"] is None and rec["end_pending"] == "terminate" and f.end_pending() == ["pod0"]
    usd0, burn = f.spend()
    assert burn == 1.0  # still counted against --max-usd
    assert "STILL BILLED" in fleet.status_text(f.state, clock.t)
    ops.fails = 0
    clock.t += 120
    f.api_tick()
    assert ("terminate", "pod0") in ops.events and f.end_pending() == []
    assert rec["ended_at"] == clock.t and f.spend()[1] == 0.0


def test_pending_terminate_confirmed_by_pod_list(tmp_path):
    clock = Clock()

    class LostReply(FakeOps):
        def terminate(self, pod_id):
            super().terminate(pod_id)  # it worked, but the reply was lost
            raise fleet.ApiFailure("timeout")

    ops = LostReply(clock, setup_polls=1, per_poll=4)
    f = make_fleet(tmp_path, clock, ops)
    f.add_batches([make_batch(tmp_path, "b1", 2)])
    drive(f, clock)  # its api_tick: pod-list shows the pod terminated, no second terminate call
    assert f.end_pending() == [] and len([e for e in ops.events if e[0] == "terminate"]) == 1
    assert "confirmed by pod-list" in (tmp_path / "fleet" / "fleet.log").read_text()


def test_drain_ends_a_pod_whose_pull_never_verifies(tmp_path):
    clock = Clock()

    class NoPull(FakeOps):
        def pull_batch(self, ep, job, name, dest):
            self.ev("pull", ep["pod_id"], name)  # copies nothing: verification keeps failing

    ops = NoPull(clock, setup_polls=1, per_poll=4)
    f = make_fleet(tmp_path, clock, ops, max_creates=1, drain_timeout_min=30)
    f.add_batches([make_batch(tmp_path, "b1", 2)])
    drive(f, clock)
    log = (tmp_path / "fleet" / "fleet.log").read_text()
    assert "pull_verify_failed" in log and "drain_unpulled" in log
    assert ("terminate", "pod0") in ops.events
    assert f.state["batches"]["b1"]["status"] == "queued"  # requeued, not marked done


def test_drain_notices_an_unreachable_pod(tmp_path):
    clock = Clock()

    class DrainDies(FakeOps):
        def pull_batch(self, ep, job, name, dest):
            self.unreachable.add(ep["pod_id"])  # the pod dies during the final pull
            raise fleet.SshError("connection reset")

    ops = DrainDies(clock, setup_polls=1, per_poll=4)
    f = make_fleet(tmp_path, clock, ops, max_creates=1, dead_after_min=5, drain_timeout_min=10**6)
    f.add_batches([make_batch(tmp_path, "b1", 2)])
    drive(f, clock)
    assert "unreachable for" in (tmp_path / "fleet" / "fleet.log").read_text()
    assert ("terminate", "pod0") in ops.events and f.state["batches"]["b1"]["status"] == "queued"


def test_replacement_pod_does_not_inherit_drain_timer(tmp_path):
    clock = Clock()

    class DrainDiesOnce(FakeOps):
        calls = 0

        def pull_batch(self, ep, job, name, dest):
            if ep["pod_id"] == "pod0":
                self.calls += 1
                if self.calls <= 2:
                    return  # copies nothing: verification fails (run, then idle), the pod goes to drain
                self.unreachable.add("pod0")  # ... and dies there
                raise fleet.SshError("connection reset")
            super().pull_batch(ep, job, name, dest)

    ops = DrainDiesOnce(clock, setup_polls=1, per_poll=4)
    f = make_fleet(tmp_path, clock, ops, max_creates=2, dead_after_min=5, drain_timeout_min=30)
    f.add_batches([make_batch(tmp_path, "b1", 2)])
    f.ensure_slots()
    seen_drain = False
    for _ in range(400):
        f.api_tick()
        f.step("s0")
        clock.t += 60
        if f.state["slots"]["s0"]["phase"] == "drain":
            seen_drain = True
        if f.state["slots"]["s0"].get("pod_id") == "pod1":
            break
    assert seen_drain
    assert f.state["slots"]["s0"]["drain_since"] is None and f.state["slots"]["s0"]["clips_prev"] == 0
    drive(f, clock)
    assert f.state["batches"]["b1"]["status"] == "done"
    assert "drain_unpulled" not in (tmp_path / "fleet" / "fleet.log").read_text()


# --------------------------------------------------------------------------- resume

def test_resume_uses_stored_config_and_guards_pod_fields(tmp_path, capsys, monkeypatch):
    clock = Clock()
    ops = FakeOps(clock, setup_polls=1, per_poll=1)
    fd = tmp_path / "fleet"
    f = fleet.Fleet(cfg(pods=1, max_usd=40.0), ops, fd, now=clock, echo=False)
    f.add_batches([make_batch(tmp_path, "b1", 8), make_batch(tmp_path, "b2", 4)])
    with f.lock:
        f.state["config"] = fleet.asdict(f.cfg)
        f.save()
    f.ensure_slots()
    for _ in range(6):  # pod0 created, set up, b1 running
        f.api_tick()
        f.step("s0")
        clock.t += 60
    assert f.state["slots"]["s0"]["phase"] == "run" and f.state["pods"]["pod0"]["ended_at"] is None

    parse = lambda argv: fleet.build_parser().parse_args(argv)  # noqa: E731
    state = fleet.Store(fd / "state.json").load()
    c, changed, problems = fleet.resolve_config(parse(["run", "--max-usd", "80"]), state)
    assert (c.gpu_type, c.max_usd, c.pods, changed, problems) == ("L40S", 80.0, 1, ["max_usd"], [])
    _, _, problems = fleet.resolve_config(parse(["run", "--gpu-type", "4090"]), state)
    assert any("--gpu-type while fleet pods are live (pod0)" in p for p in problems)
    _, _, problems = fleet.resolve_config(parse(["run", "--video-precision", "fp32"]), state)
    assert any("mix outputs" in p for p in problems)

    # through main(): a changed pod field is refused before anything runs ...
    monkeypatch.setattr(fleet.signal, "signal", lambda *a: None)
    runs = []
    monkeypatch.setattr(fleet.Fleet, "run", lambda self: runs.append(self) or 0)
    assert fleet.main(["run", "--go", "--gpu-type", "4090"], ops=ops, fleet_dir=fd) == 2
    assert "refused" in capsys.readouterr().err and not runs
    # ... a plain `run --go` resumes with the stored config (preflight included) ...
    with f.lock:
        f.state["halt"] = {"reason": "test", "at": clock.t}
        f.save()
    assert fleet.main(["run", "--go", "--pods", "2"], ops=ops, fleet_dir=fd) == 0
    g = runs[0]
    assert (g.cfg.gpu_type, g.cfg.max_usd, g.cfg.pods) == ("L40S", 40.0, 2)
    assert g.state["halt"] is None  # a restart clears a halt
    assert g.state["order"] == ["b1", "b2"]  # no default batches mixed into a resumed queue
    # ... and the plan says so
    assert fleet.main(["run", "--max-usd", "60"], ops=RaisingOps(), fleet_dir=fd) == 0
    assert "resuming fleet run" in capsys.readouterr().out

    # a second driver instance continues the in-flight pod from state.json alone
    h = fleet.Fleet(fleet.stored_config(fleet.Store(fd / "state.json").load()), ops, fd, now=clock, echo=False)
    drive(h, clock)
    # --pods 2 was stored: s1's pod found nothing left while setting up and was ended at once
    assert [e for e in ops.events if e[0] == "create"] == [("create", "pod0"), ("create", "pod1")]
    assert ("terminate", "pod1") in ops.events and "pod1" not in {e[1] for e in ops.events if e[0] == "push"}
    assert all(b["status"] == "done" for b in h.state["batches"].values())
    assert fleet.local_done_ids(fd / "outputs" / "b1") == {f"b1v{i:03d}" for i in range(8)}
    assert [e for e in ops.events if e[0] == "push" and e[2] == "b1"] == [("push", "pod0", "b1")]

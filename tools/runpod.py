#!/usr/bin/env python3
"""RunPod from the command line: GPU prices/stock, pods, volumes, spend watchdog.

Replaces the console steps in docs/RUNPOD.md. Stdlib only. Cheap default path:
community cloud, on-demand, a pod volume at /workspace (deleted on terminate).

    tools/runpod.py gpus                                         # cheapest first: price, stock, VRAM, RAM
    tools/runpod.py watchdog --max-usd 10 --max-hours 6          # start FIRST, under nohup
    tools/runpod.py pod-create --name tribe-p1 --gpu-type 4090 --max-hours 3
    eval "$(tools/runpod.py pod-wait POD_ID)"                    # sets POD / POD_PORT for pod.sh
    tools/runpod.py pod-list | pod-stop ID | pod-start ID | pod-terminate ID [--yes] | balance
    tools/runpod.py volume-create/volume-list                    # optional network volume (Secure)

Secrets (RUNPOD_API_KEY, HF_TOKEN) come from .env in the repo root and are never
printed: every line of output is scrubbed of .env values and pod `env` blocks
are redacted. All network traffic goes through `_http` (tests replace it).

APIs: REST v1 (https://rest.runpod.io/v1, spec at /v1/openapi.json) for pods and
volumes; GraphQL (https://api.runpod.io/graphql) only where REST has no field:
GPU stock per datacenter, account balance and live GPU utilisation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
STATE_FILE = ROOT / "results" / "runpod_state.json"
HEARTBEAT_FILE = ROOT / "results" / "runpod_watchdog.json"
PUBKEY_FILE = Path.home() / ".ssh" / "id_ed25519.pub"

REST_URL = "https://rest.runpod.io/v1"
GRAPHQL_URL = "https://api.runpod.io/graphql"
# Documented v2 catalog (docs.runpod.io/api-reference-v2/catalog/list-gpu-types):
# list price per GPU per hour and availability for a pod of `count` GPUs.
CATALOG_URL = "https://api.runpod.io/v2/catalog/gpus"
HTTP_TIMEOUT = 30
USER_AGENT = "tribe-research-runpod/1.0"

PREFIX = "tribe-"  # the watchdog only ever touches pods with this name prefix
# Python 3.11 + CUDA 12.4, as setup.sh expects. The only runpod/pytorch tag on
# Docker Hub naming both (digest sha256:61a4aafb0094cd77…, ~7.4 GB compressed).
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
MOUNT_PATH = "/workspace"
# Host drivers (REST allowedCudaVersions enum). The image is CUDA 12.4, but whisperx 3.8.6
# (pod/whisper_server.py and TRIBE's own `uvx whisperx`) runs torch 2.8 wheels built for
# CUDA 12.8, so an older driver would fail at transcription after setup was paid for.
CUDA_VERSIONS = ["12.4", "12.5", "12.6", "12.7", "12.8", "12.9", "13.0"]
DEFAULT_MIN_CUDA = "12.8"
# torch 2.6 in setup.sh has no Blackwell kernels; setup.sh refuses these anyway.
BLACKWELL_MARKERS = ("B200", "B300", "BLACKWELL", "RTX 5060", "RTX 5070", "RTX 5080", "RTX 5090")
# Snapshot of the REST gpuTypeIds enum (openapi.json, 2026-09-26).
GPU_TYPE_IDS = [
    "AMD Instinct MI300X OAM", "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100-SXM4-80GB", "NVIDIA A40", "NVIDIA B200", "NVIDIA B300 SXM6 AC",
    "NVIDIA B300 SXM6 AC MIG 1g.34gb", "NVIDIA GeForce RTX 3070", "NVIDIA GeForce RTX 3080",
    "NVIDIA GeForce RTX 3080 Ti", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA GeForce RTX 4070 Ti", "NVIDIA GeForce RTX 4080", "NVIDIA GeForce RTX 4080 SUPER",
    "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5080", "NVIDIA GeForce RTX 5090",
    "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA H200",
    "NVIDIA H200 NVL", "NVIDIA L4", "NVIDIA L40", "NVIDIA L40S",
    "NVIDIA RTX 2000 Ada Generation", "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX 4000 SFF Ada Generation", "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX A2000", "NVIDIA RTX A4000",
    "NVIDIA RTX A4500", "NVIDIA RTX A5000", "NVIDIA RTX A6000",
    "NVIDIA RTX PRO 4000 Blackwell", "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA RTX PRO 5000 Blackwell", "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "Tesla V100-PCIE-16GB",
    "Tesla V100-SXM2-16GB",
]


class ApiError(Exception):
    pass


class UsageError(Exception):
    pass


# --------------------------------------------------------------------------- secrets

_SECRETS: set[str] = set()
_ENV_CACHE: dict[str, str] | None = None


def parse_env(text: str) -> dict[str, str]:
    """KEY=VALUE lines; blank lines, comments and `export ` prefixes allowed."""
    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            env[key] = value
    return env


def load_env(path: Path | None = None) -> dict[str, str]:
    """Read .env once; every value long enough to be a credential is scrubbed from output."""
    global _ENV_CACHE
    if _ENV_CACHE is None or path is not None:
        p = path or ENV_FILE
        env = parse_env(p.read_text()) if p.exists() else {}
        for v in env.values():
            if len(v) >= 8:
                _SECRETS.add(v)
        _ENV_CACHE = env
    return _ENV_CACHE


def need(key: str) -> str:
    value = load_env().get(key, "")
    if not value:
        raise UsageError(f"{key} missing from {ENV_FILE} (KEY=VALUE line)")
    return value


def scrub(text: str) -> str:
    for s in sorted(_SECRETS, key=len, reverse=True):
        text = text.replace(s, "***")
    return text


_REDACT_KEYS = {"env", "apikey", "pw", "password", "token", "secret"}


def redact(obj):
    """Copy of an API object with env values and credential-looking fields masked."""
    if isinstance(obj, dict):
        outd = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if lk == "env" and isinstance(v, dict):
                outd[k] = {ek: "***" for ek in v}
            elif lk == "env" and isinstance(v, list):  # GraphQL shape: ["K=V", ...]
                outd[k] = [str(e).split("=", 1)[0] + "=***" for e in v]
            elif lk in _REDACT_KEYS and v not in (None, "", [], {}):
                outd[k] = "***"
            else:
                outd[k] = redact(v)
        return outd
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return scrub(obj)
    return obj


def out(*parts) -> None:
    print(scrub(" ".join(str(p) for p in parts)), flush=True)


def err(*parts) -> None:
    print(scrub(" ".join(str(p) for p in parts)), file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- transport

def _http(method: str, url: str, body=None) -> tuple[int, bytes]:
    """The only function that touches the network. Headers never leave it."""
    headers = {
        "Authorization": "Bearer " + need("RUNPOD_API_KEY"),
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        raise ApiError(f"{method} {urllib.parse.urlsplit(url).path}: network error: {reason}") from None


def rest(method: str, path: str, body=None, params: dict | None = None):
    url = REST_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    status, raw = _http(method, url, body)
    if status >= 400:
        raise ApiError(f"{method} {path}: HTTP {status}: {scrub(raw[:500].decode(errors='replace'))}")
    if not raw or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except ValueError:
        raise ApiError(f"{method} {path}: HTTP {status}: non-JSON response") from None


def graphql(query: str, variables: dict | None = None) -> dict:
    status, raw = _http("POST", GRAPHQL_URL, {"query": query, "variables": variables or {}})
    text = raw.decode(errors="replace")
    if status >= 400:
        raise ApiError(f"graphql: HTTP {status}: {scrub(text[:500])}")
    try:
        doc = json.loads(raw)
    except ValueError:
        raise ApiError(f"graphql: HTTP {status}: non-JSON response") from None
    if doc.get("errors"):
        raise ApiError("graphql: " + scrub(str(doc["errors"][0].get("message", doc["errors"][0]))))
    return doc.get("data") or {}


# --------------------------------------------------------------------------- helpers

def now_ts() -> float:
    return time.time()


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value) -> float | None:
    """RFC3339 / ISO strings or unix seconds/ms (as number or string) -> epoch seconds."""
    if value in (None, ""):
        return None
    try:
        n = float(value)
        return n / 1000 if n > 1e12 else n
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
            if isinstance(state, dict):
                state.setdefault("pods", {})
                return state
        except ValueError:
            err(f"warning: {STATE_FILE} unreadable; treating as empty")
    return {"pods": {}}


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def resolve_gpu(query: str) -> str:
    """'L40S' -> 'NVIDIA L40S'. Exact id, then unique suffix, then unique substring."""
    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    q = norm(query)
    for g in GPU_TYPE_IDS:
        if norm(g) == q:
            return g
    for match in (lambda g: norm(g).endswith(q), lambda g: q in norm(g)):
        hits = [g for g in GPU_TYPE_IDS if match(g)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise UsageError(f"--gpu-type {query!r} is ambiguous: " + "; ".join(hits))
    if query.startswith(("NVIDIA ", "AMD ", "Tesla ")):
        err(f"warning: {query!r} is not in this script's GPU list; passing it through as-is")
        return query
    raise UsageError(f"unknown --gpu-type {query!r}. Known: " + "; ".join(GPU_TYPE_IDS))


def is_blackwell(gpu_id: str) -> bool:
    return any(m in gpu_id.upper() for m in BLACKWELL_MARKERS)


def ssh_endpoint(pod: dict) -> tuple[str, int] | None:
    """(public IP, public port for 22) once the pod exposes full SSH, else None."""
    ip = pod.get("publicIp") or ""
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22", mappings.get(22)) if isinstance(mappings, dict) else None
    if ip and port:
        return ip, int(port)
    return None


def is_terminated(pod: dict) -> bool:
    # A terminate writes desiredStatus EXITED + "Terminated by ..." (runpodctl podstate notes).
    return (pod.get("desiredStatus") == "TERMINATED"
            or "terminated" in str(pod.get("lastStatusChange") or "").lower())


def pod_gpu_name(pod: dict) -> str:
    gpu = pod.get("gpu") or {}
    machine = pod.get("machine") or {}
    return gpu.get("displayName") or gpu.get("id") or machine.get("gpuDisplayName") or "?"


# --------------------------------------------------------------------------- gpus

# runpodctl-proven query; the spot-price variant below falls back to it on error.
Q_GPU_TYPES = """query { gpuTypes { id displayName memoryInGb secureCloud communityCloud
  securePrice communityPrice } }"""
Q_GPU_TYPES_SPOT = """query { gpuTypes { id displayName memoryInGb secureCloud communityCloud
  securePrice communityPrice secureSpotPrice communitySpotPrice } }"""
Q_DATACENTERS = """query { dataCenters { id name location
  gpuAvailability { gpuTypeId displayName stockStatus } } }"""
# From the GraphQL spec (graphql-spec.runpod.io), not used by runpodctl: price,
# stock and host RAM among hosts that can take this pod. On error: stock only.
Q_PRICE_DETAIL = """query PriceDetail($n: Int!, $ram: Int, $dc: String) {
  dataCenters { id storageSupport }
  gpuTypes { id
    sec: lowestPrice(input: {gpuCount: $n, secureCloud: true, minMemoryInGb: $ram,
                             dataCenterId: $dc, supportPublicIp: true}) { ...P }
    com: lowestPrice(input: {gpuCount: $n, secureCloud: false, minMemoryInGb: $ram,
                             dataCenterId: $dc, supportPublicIp: true}) { ...P } } }
fragment P on LowestPrice { stockStatus uninterruptablePrice minimumBidPrice minMemory
  maxUnreservedGpuCount }"""

# Always listed so the cheap candidates can be compared even when out of stock.
SHORTLIST = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA A40",
             "NVIDIA RTX A6000", "NVIDIA L40S", "NVIDIA L4"]
STOCK_RANK = {"high": 3, "medium": 2, "low": 1}


def stock_rank(s) -> int:
    return STOCK_RANK.get(str(s or "").strip().lower(), 0)


def short_gpu(gid: str) -> str:
    return gid.replace("NVIDIA ", "").replace("GeForce ", "")


def money(v) -> str:
    f = to_float(v, 0.0)
    return f"{f:.2f}" if f > 0 else "-"


def catalog(count: int, cloud: str = "COMMUNITY") -> dict[str, dict]:
    """{gpu type id: catalog entry} from the v2 catalog, with availability for ``count`` GPUs."""
    q = urllib.parse.urlencode({"include": "AVAILABILITY", "product": "POD", "count": count, "cloud": cloud})
    status, raw = _http("GET", f"{CATALOG_URL}?{q}")
    if status >= 400:
        raise ApiError(f"catalog: HTTP {status}: {scrub(raw[:300].decode(errors='replace'))}")
    try:
        doc = json.loads(raw)
    except ValueError:
        raise ApiError("catalog: non-JSON response") from None
    items = doc if isinstance(doc, list) else next(
        (doc[k] for k in ("data", "gpus", "items", "gpuTypes") if isinstance(doc.get(k), list)), [])
    return {g.get("id") or g.get("gpuTypeId"): g for g in items if isinstance(g, dict)
            and (g.get("id") or g.get("gpuTypeId"))}


def cmd_gpus(args) -> int:
    try:
        types = graphql(Q_GPU_TYPES_SPOT).get("gpuTypes") or []
    except ApiError as e:
        err(f"note: spot prices unavailable ({e})")
        types = graphql(Q_GPU_TYPES).get("gpuTypes") or []
    dcs = graphql(Q_DATACENTERS).get("dataCenters") or []
    if args.dc and not any(dc.get("id") == args.dc for dc in dcs):
        err(f"warning: datacenter {args.dc} not in the API's list: " + ", ".join(sorted(str(d.get("id")) for d in dcs)))
    where_by_type: dict[str, list[tuple[str, str]]] = {}
    for dc in dcs:
        if args.dc and dc.get("id") != args.dc:
            continue
        for a in dc.get("gpuAvailability") or []:
            if stock_rank(a.get("stockStatus")):
                where_by_type.setdefault(a.get("gpuTypeId"), []).append((dc.get("id"), a.get("stockStatus")))

    detail: dict[str, dict] = {}
    storage: dict[str, object] = {}
    ram_total = args.min_ram_gb * args.gpu_count
    try:
        d = graphql(Q_PRICE_DETAIL, {"n": args.gpu_count, "ram": ram_total, "dc": args.dc})
        detail = {t.get("id"): t for t in d.get("gpuTypes") or []}
        storage = {x.get("id"): x.get("storageSupport") for x in d.get("dataCenters") or []}
    except ApiError as e:
        err(f"note: per-host price/RAM query failed ({e}); showing list prices, "
            "and stock is not filtered by host RAM")

    cat: dict[str, dict] = {}
    try:
        cat = catalog(args.gpu_count)
    except ApiError as e:
        err(f"note: v2 catalog unavailable ({e}); no availability for {args.gpu_count} GPU/pod")

    rows = []
    for t in types:
        gid = t.get("id")
        if not gid or gid == "unknown":
            continue
        det = detail.get(gid) or {}
        sec, com = det.get("sec") or {}, det.get("com") or {}
        where = sorted(where_by_type.get(gid, []), key=lambda w: (-stock_rank(w[1]), w[0]))
        in_stock = bool(where) or stock_rank(sec.get("stockStatus")) or stock_rank(com.get("stockStatus"))
        vram = to_float(t.get("memoryInGb"))
        if not args.all and gid not in SHORTLIST and (vram < args.min_vram_gb or not in_stock):
            continue
        c = cat.get(gid) or {}
        # lowestPrice is for the whole pod (live: 4x A100 PCIe = 4.76 = 4 x 1.19); show per GPU
        per_gpu = lambda v: to_float(v) / args.gpu_count if to_float(v) > 0 else None  # noqa: E731
        p = {
            "com": per_gpu(com.get("uninterruptablePrice")) or (c.get("price") or {}).get("community")
            or (t.get("communityPrice") if t.get("communityCloud") else None),
            "com_spot": per_gpu(com.get("minimumBidPrice")) or t.get("communitySpotPrice"),
            "sec": per_gpu(sec.get("uninterruptablePrice"))
            or (t.get("securePrice") if t.get("secureCloud") else None),
            "sec_spot": per_gpu(sec.get("minimumBidPrice")) or t.get("secureSpotPrice"),
        }
        on_demand = [to_float(v) for v in (p["com"], p["sec"]) if to_float(v) > 0]
        p["avail"] = str(c.get("availability") or "-") if cat else "?"
        rows.append((min(on_demand) if on_demand else 9e9, gid, vram, p, sec, com, where))
    rows.sort(key=lambda r: (r[0], r[1]))

    scope = f"datacenter {args.dc}" if args.dc else "all datacenters"
    out(f"{scope}; {args.gpu_count} GPU/pod; host RAM >= {args.min_ram_gb} GB/GPU "
        f"({ram_total} GB asked); $/h per GPU, on-demand and spot (bid)"
        + (f"; network volumes here: {'yes' if storage.get(args.dc) else 'no'}" if args.dc in storage else ""))
    out(f"{'GPU':<30} {'VRAM':>4} {'COM':>5} {'spot':>5} {'SEC':>5} {'spot':>5} "
        f"{'stock com/sec':>14} {'RAM':>4} {'free':>4} {f'x{args.gpu_count}':>6}  datacenters with stock")
    for _, gid, vram, p, sec, com, where in rows:
        stock = f"{com.get('stockStatus') or '-'}/{sec.get('stockStatus') or '-'}" if (sec or com) else "?"
        rams = [to_float(x.get("minMemory")) for x in (com, sec) if x.get("minMemory")]
        ram = f"{min(rams):.0f}" if rams else "?"
        frees = [x.get("maxUnreservedGpuCount") for x in (com, sec) if x.get("maxUnreservedGpuCount") is not None]
        free = str(max(frees)) if frees else "?"
        flag = "" if vram >= args.min_vram_gb else "  (VRAM too small for TRIBE)"
        dcs_txt = " ".join(f"{dc}:{s}" for dc, s in where) or "-"
        out(f"{short_gpu(gid):<30} {vram:>4.0f} {money(p['com']):>5} {money(p['com_spot']):>5} "
            f"{money(p['sec']):>5} {money(p['sec_spot']):>5} {stock:>14} {ram:>4} {free:>4} {p['avail']:>6}"
            f"  {dcs_txt}{flag}")
    out("COM = community cloud (cheapest; pod volume only), SEC = secure cloud (network volumes). "
        "RAM = least host RAM (GB) the API reports for a matching host; ? = not reported. "
        f"x{args.gpu_count} = community availability for a {args.gpu_count}-GPU pod (v2 catalog).")
    return 0


# --------------------------------------------------------------------------- volumes

def cmd_volume_create(args) -> int:
    if args.size_gb < 10:
        raise UsageError("--size-gb must be at least 10")
    vol = rest("POST", "/networkvolumes", {"name": args.name, "size": args.size_gb, "dataCenterId": args.dc})
    vol = vol or {}
    out(f"created network volume {vol.get('id')} ({vol.get('name')}, {vol.get('size')} GB, {vol.get('dataCenterId')})")
    out("It bills monthly until deleted, whether or not a pod runs.")
    return 0


def cmd_volume_list(args) -> int:
    vols = rest("GET", "/networkvolumes") or []
    out(f"{'id':<14} {'size GB':>7} {'datacenter':<10} name")
    for v in vols:
        out(f"{v.get('id', ''):<14} {v.get('size', ''):>7} {v.get('dataCenterId', ''):<10} {v.get('name', '')}")
    return 0


# --------------------------------------------------------------------------- pods

def build_pod_payload(*, name: str, gpu_type: str, gpu_count: int, cloud: str, image: str,
                      container_disk_gb: int, public_key: str, hf_token: str,
                      volume_id: str | None = None, pod_volume_gb: int = 120, dc: str | None = None,
                      min_ram_gb: int = 48, interruptible: bool = False,
                      min_cuda: str = DEFAULT_MIN_CUDA) -> dict:
    payload = {
        "name": name,
        "imageName": image,
        "computeType": "GPU",
        "cloudType": cloud,
        "interruptible": bool(interruptible),  # spot only when asked for explicitly
        "gpuTypeIds": [gpu_type],
        "gpuCount": gpu_count,
        "minRAMPerGPU": min_ram_gb,  # host RAM per GPU; TRIBE's model stack is RAM-hungry
        "volumeMountPath": MOUNT_PATH,
        "containerDiskInGb": container_disk_gb,
        "ports": ["22/tcp"],  # full SSH only; the REST default would also publish Jupyter
        "supportPublicIp": True,  # community hosts without a public IP cannot expose TCP
        "allowedCudaVersions": [v for v in CUDA_VERSIONS
                                if tuple(map(int, v.split("."))) >= tuple(map(int, min_cuda.split(".")))],
        "env": {"PUBLIC_KEY": public_key, "HF_TOKEN": hf_token},
    }
    if volume_id:
        payload["networkVolumeId"] = volume_id
    else:
        payload["volumeInGb"] = pod_volume_gb  # pod volume at /workspace; deleted on terminate
    if dc:
        payload["dataCenterIds"] = [dc]  # a network volume only mounts in its own DC
    return payload


def watchdog_fresh(now: float) -> tuple[bool, str]:
    if not HEARTBEAT_FILE.exists():
        return False, "no watchdog heartbeat"
    try:
        hb = json.loads(HEARTBEAT_FILE.read_text())
    except ValueError:
        return False, "watchdog heartbeat unreadable"
    age = now - to_float(hb.get("epoch"))
    limit = 3 * to_float(hb.get("interval"), 60) + 60
    if age > limit:
        return False, f"watchdog heartbeat is {age / 60:.0f} min old"
    return True, f"watchdog alive (caps ${hb.get('max_usd')}, {hb.get('max_hours')} h)"


def cmd_pod_create(args) -> int:
    if not args.name.startswith(PREFIX):
        raise UsageError(f"--name must start with {PREFIX!r} so the watchdog can see it")
    if args.max_hours <= 0:
        raise UsageError("--max-hours must be > 0")
    if args.volume_id and args.cloud != "SECURE":
        raise UsageError("network volumes exist only in Secure Cloud: add --cloud SECURE, "
                         "or drop --volume-id to use a pod volume")
    gpu = resolve_gpu(args.gpu_type)
    if is_blackwell(gpu):
        raise UsageError(f"{gpu} is Blackwell; torch 2.6 in setup.sh does not support it")
    hf_token = need("HF_TOKEN")
    pubkey_path = Path(args.public_key_file).expanduser()
    if not pubkey_path.exists():
        raise UsageError(f"{pubkey_path} not found")
    public_key = pubkey_path.read_text().strip()
    if not public_key.startswith("ssh-"):
        raise UsageError(f"{pubkey_path} does not look like an OpenSSH public key")

    dc = args.dc
    if not args.dry_run:
        ok, why = watchdog_fresh(now_ts())
        if not ok and not args.allow_no_watchdog:
            raise UsageError(f"{why}. Start it first: mkdir -p results; nohup tools/runpod.py watchdog "
                             f"--max-usd 10 --max-hours 6 >> results/watchdog.log 2>&1 &  "
                             f"(or --allow-no-watchdog)")
        err(why)
        if args.volume_id:
            vol = rest("GET", f"/networkvolumes/{args.volume_id}") or {}
            dc = vol.get("dataCenterId") or dc
            if not dc:
                raise UsageError(f"network volume {args.volume_id}: no dataCenterId in the API response")
            err(f"volume {args.volume_id} is in {dc}")

    payload = build_pod_payload(name=args.name, gpu_type=gpu, gpu_count=args.gpu_count, cloud=args.cloud,
                                image=args.image, container_disk_gb=args.container_disk_gb,
                                public_key=public_key, hf_token=hf_token, volume_id=args.volume_id,
                                pod_volume_gb=args.volume_gb, dc=dc, min_ram_gb=args.min_ram_gb,
                                interruptible=args.interruptible, min_cuda=args.min_cuda)
    if args.dry_run:
        out(json.dumps(redact(payload), indent=2))
        out("(dry run: nothing sent)")
        return 0

    created_at = now_ts()
    try:
        pod = rest("POST", "/pods", payload) or {}
    except ApiError as e:
        # Never retry a create: a timeout may still have made a billed pod.
        err(f"pod-create failed or outcome unknown: {e}")
        err("Run `tools/runpod.py pod-list` before trying again; the watchdog acts on any "
            f"{PREFIX} pod past its caps.")
        return 1
    pid = pod.get("id")
    if not pid:
        err("pod-create: response had no pod id; run `tools/runpod.py pod-list`")
        err(json.dumps(redact(pod))[:1000])
        return 1
    state = load_state()
    state["pods"][pid] = {
        "name": args.name,
        "created_at": iso(created_at),
        "max_hours": args.max_hours,
        "cost_per_hr": to_float(pod.get("costPerHr"), 0.0) or None,
        "gpu_type": gpu,
        "gpu_count": args.gpu_count,
        "cloud": args.cloud,
        "interruptible": bool(args.interruptible),
        "volume_id": args.volume_id,
        "pod_volume_gb": None if args.volume_id else args.volume_gb,
        "datacenter": dc,
        "image": args.image,
    }
    write_json_atomic(STATE_FILE, state)
    out(f"created pod {pid} ({args.name}): {args.gpu_count}x {gpu}, {args.cloud}"
        f"{' SPOT' if args.interruptible else ''}{' in ' + dc if dc else ''}, "
        f"${pod.get('costPerHr', '?')}/h, max {args.max_hours} h")
    if not args.volume_id:
        out(f"/workspace is a {args.volume_gb} GB POD volume: terminating the pod DELETES it. "
            "Pull results before pod-terminate.")
    out(f"next: eval \"$(tools/runpod.py pod-wait {pid})\"")
    return 0


# Runtime ports as runpodctl reads them; used when REST has no portMappings yet.
Q_RUNTIME_PORTS = """query { myself { pods { id
  runtime { ports { ip isIpPublic privatePort publicPort type } } } } }"""


def ssh_from_runtime(pod_id: str) -> tuple[str, int] | None:
    try:
        me = graphql(Q_RUNTIME_PORTS).get("myself") or {}
    except ApiError:
        return None
    for p in me.get("pods") or []:
        if p.get("id") != pod_id:
            continue
        for port in ((p.get("runtime") or {}).get("ports") or []):
            if (port.get("privatePort") == 22 and port.get("isIpPublic") and port.get("ip")
                    and port.get("publicPort") and str(port.get("type", "tcp")).lower() == "tcp"):
                return port["ip"], int(port["publicPort"])
    return None


def cmd_pod_wait(args) -> int:
    deadline = now_ts() + args.timeout_min * 60
    last = None
    while True:
        try:
            pod = rest("GET", f"/pods/{args.pod_id}") or {}
        except ApiError as e:
            err(f"poll failed: {e}")
            pod = None
        if pod:
            if is_terminated(pod) or pod.get("desiredStatus") == "EXITED":
                err(f"pod {args.pod_id} is {pod.get('desiredStatus')}: {pod.get('lastStatusChange', '')}")
                return 1
            ep = None
            if pod.get("desiredStatus") == "RUNNING":
                ep = ssh_endpoint(pod) or ssh_from_runtime(args.pod_id)
            if ep:
                ip, port = ep
                err(f"pod {args.pod_id} up: ssh root@{ip} -p {port}")
                print(f"export POD=root@{ip} POD_PORT={port}", flush=True)
                return 0
            status = f"{pod.get('desiredStatus')} ip={pod.get('publicIp') or '-'}"
            if status != last:
                err(f"waiting: {status}")
                last = status
        if now_ts() >= deadline:
            err(f"timed out after {args.timeout_min} min; the pod is still billed. "
                f"Check pod-list or pod-terminate {args.pod_id}")
            return 1
        _sleep(args.interval)


def cmd_pod_list(args) -> int:
    pods = rest("GET", "/pods") or []
    if args.json:
        out(json.dumps(redact(pods), indent=2))
        return 0
    state = load_state()["pods"]
    now = now_ts()
    out(f"{'id':<16} {'name':<22} {'status':<8} {'$/h':>6} {'age h':>6}  gpu / ssh")
    for p in pods:
        start = earliest(state.get(p.get("id"), {}).get("created_at"), p.get("createdAt"))
        age = f"{(now - start) / 3600:.1f}" if start else "?"
        ep = ssh_endpoint(p)
        ssh = f"root@{ep[0]} -p {ep[1]}" if ep else "-"
        out(f"{p.get('id', ''):<16} {p.get('name', ''):<22} {p.get('desiredStatus', ''):<8} "
            f"{to_float(p.get('costPerHr')):>6.2f} {age:>6}  {pod_gpu_name(p)} / {ssh}")
    return 0


def has_network_volume(pod: dict, rec: dict | None = None) -> bool:
    return bool(pod.get("networkVolumeId") or (pod.get("networkVolume") or {}).get("id")
                or (rec or {}).get("volume_id"))


def _mark(pod_id: str, key: str, now: float) -> None:
    state = load_state()
    if pod_id in state["pods"]:
        state["pods"][pod_id][key] = iso(now)
        write_json_atomic(STATE_FILE, state)


def terminate_pod(pod_id: str, now: float) -> None:
    rest("DELETE", f"/pods/{pod_id}")
    _mark(pod_id, "ended_at", now)


def stop_pod(pod_id: str, now: float) -> None:
    rest("POST", f"/pods/{pod_id}/stop")
    _mark(pod_id, "stopped_at", now)


POD_VOLUME_WARNING = ("WARNING: this pod has no network volume. Terminating DELETES its /workspace "
                      "pod volume: outputs, logs, venv and weights. Pull first: tools/pod.sh pull <run>.")


def record_resume(rec: dict, now: float) -> None:
    """A restarted pod: the watchdog times this run from `resumed_at` (not creation) and adds what the
    earlier runs cost (`spent_before_usd`, from the recorded start/stop times)."""
    run_start = parse_time(rec.get("resumed_at")) or parse_time(rec.get("created_at"))
    stopped = parse_time(rec.get("stopped_at"))
    if run_start is not None and stopped is not None and stopped > run_start:
        prior = to_float(rec.get("spent_before_usd")) + to_float(rec.get("cost_per_hr")) * (stopped - run_start) / 3600
        rec["spent_before_usd"] = round(prior, 4)
    rec["resumed_at"] = iso(now)
    rec.pop("stopped_at", None)


def cmd_pod_stop(args) -> int:
    stop_pod(args.pod_id, now_ts())
    out(f"stopped {args.pod_id}. GPU billing stops; /workspace (pod or network volume) is kept, and a "
        f"pod volume is still billed for storage. Restarting needs a free GPU on the same host.")
    return 0


def cmd_pod_start(args) -> int:
    state = load_state()
    rec = state["pods"].setdefault(args.pod_id, {})
    now = now_ts()
    record_resume(rec, now)
    if args.grace_min > 0:
        rec["grace_until"] = iso(now + args.grace_min * 60)
    write_json_atomic(STATE_FILE, state)
    rest("POST", f"/pods/{args.pod_id}/start")
    out(f"started {args.pod_id}; the watchdog leaves it alone for {args.grace_min:g} min "
        f"(still counted in spend). Then: eval \"$(tools/runpod.py pod-wait {args.pod_id})\"")
    return 0


def cmd_pod_terminate(args) -> int:
    pod = rest("GET", f"/pods/{args.pod_id}") or {}
    rec = load_state()["pods"].get(args.pod_id, {})
    if not has_network_volume(pod, rec):
        err(POD_VOLUME_WARNING)
        if not args.yes:
            err("Re-run with --yes to terminate anyway.")
            return 2
    terminate_pod(args.pod_id, now_ts())
    kept = "network volume kept" if has_network_volume(pod, rec) else "pod volume deleted"
    out(f"terminated {args.pod_id} ({kept}).")
    return 0


Q_BALANCE = "query { myself { clientBalance currentSpendPerHr spendLimit } }"


def cmd_balance(args) -> int:
    me = graphql(Q_BALANCE).get("myself") or {}
    bal, rate = to_float(me.get("clientBalance")), to_float(me.get("currentSpendPerHr"))
    runway = f", runway {bal / rate:.1f} h" if rate > 0 else ""
    out(f"balance ${bal:.2f}, spending ${rate:.3f}/h{runway}, spend limit {me.get('spendLimit')}")
    return 0


# --------------------------------------------------------------------------- watchdog

Q_TELEMETRY = """query { myself { clientBalance currentSpendPerHr
  pods { id runtime { uptimeInSeconds gpus { id gpuUtilPercent } } } } }"""


def earliest(*values) -> float | None:
    ts = [t for t in (parse_time(v) for v in values) if t is not None]
    return min(ts) if ts else None


class Watchdog:
    """Pure decision logic; `tick` takes the API snapshot and a clock value.

    Action per pod: `terminate` when /workspace is a network volume (nothing is
    lost), `stop` when it is a pod volume (terminate would delete unpulled
    results; stop ends GPU billing and keeps the disk)."""

    def __init__(self, max_usd: float, max_hours: float, idle_minutes: float = 20):
        self.max_usd = max_usd
        self.max_hours = max_hours
        self.idle_s = idle_minutes * 60
        self.seen: dict[str, float] = {}      # tribe- pod still listed -> spend estimate
        self.ended: dict[str, float] = {}     # pods that vanished during this run -> final estimate
        self.first_seen: dict[str, float] = {}
        self.armed: set[str] = set()          # pods that have used the GPU at least once
        self.idle_since: dict[str, float] = {}
        self.pending: dict[str, str] = {}     # actions that failed; retried every tick

    def tick(self, pods: list[dict], telemetry: dict | None, state: dict, now: float):
        """Returns (status line, [(pod_id, action, reason)]), action in {stop, terminate}.
        `telemetry` maps pod id -> runtime, or is None when that query failed."""
        listed = [p for p in pods if str(p.get("name") or "").startswith(PREFIX) and not is_terminated(p)]
        listed_ids = {p.get("id") for p in listed}
        for pid in list(self.seen):
            if pid not in listed_ids:
                self.ended[pid] = self.seen.pop(pid)
                self.pending.pop(pid, None)
        reasons: dict[str, str] = {}
        parts = []
        running_usd = frozen_usd = rate_total = 0.0
        running = []
        for p in listed:
            pid = p.get("id")
            self.ended.pop(pid, None)  # back after a flicker in the list: count it once
            rec = (state.get("pods") or {}).get(pid, {})
            if p.get("desiredStatus") == "EXITED":
                # Stopped: no GPU billing. Keep what it cost while we watched it run.
                usd = self.seen.get(pid, 0.0)
                self.seen[pid] = usd
                frozen_usd += usd
                self.pending.pop(pid, None)
                self.armed.discard(pid)
                self.idle_since.pop(pid, None)
                parts.append(f"{pid} {p.get('name')} stopped ${usd:.2f}")
                continue
            running.append(p)
            self.first_seen.setdefault(pid, now)
            runtime = (telemetry or {}).get(pid) if telemetry is not None else None
            # a restarted pod (tools/runpod.py pod-start) is timed from its resume; earlier runs add a fixed cost
            start = parse_time(rec.get("resumed_at")) or earliest(rec.get("created_at"), p.get("createdAt"))
            prior_usd = to_float(rec.get("spent_before_usd")) if rec.get("resumed_at") else 0.0
            if start is None and runtime and runtime.get("uptimeInSeconds") is not None:
                start = now - to_float(runtime.get("uptimeInSeconds"))
            if start is None:
                start = self.first_seen[pid]
            hours = max(0.0, (now - start) / 3600)
            rate = to_float(p.get("costPerHr")) or to_float(rec.get("cost_per_hr"))
            # Errs high for a pod restarted outside pod-start (no resumed_at): billed as if it ran throughout.
            usd = max(prior_usd + rate * hours, self.seen.get(pid, 0.0))
            self.seen[pid] = usd
            running_usd += usd
            rate_total += rate
            grace = parse_time(rec.get("grace_until"))
            in_grace = grace is not None and now < grace
            cap_h = self.max_hours
            if rec.get("max_hours"):
                cap_h = min(cap_h, to_float(rec["max_hours"], cap_h))
            if hours >= cap_h and not in_grace:
                reasons[pid] = f"max-hours {cap_h:g} reached ({hours:.2f} h)"

            util_txt = "gpu=?"
            gpus = (runtime or {}).get("gpus")
            utils = [g.get("gpuUtilPercent") for g in gpus] if isinstance(gpus, list) and gpus else []
            if utils and all(u is not None for u in utils):
                util_txt = "gpu=" + "/".join(f"{int(u)}%" for u in utils)
                if any(to_float(u) > 0 for u in utils):
                    self.armed.add(pid)
                    self.idle_since.pop(pid, None)
                elif pid in self.armed:
                    since = self.idle_since.setdefault(pid, now)
                    if self.idle_s > 0 and now - since >= self.idle_s and not in_grace:
                        reasons.setdefault(pid, f"idle: GPU 0% for {(now - since) / 60:.0f} min")
            # unknown telemetry neither arms nor advances the idle clock
            parts.append(f"{pid} {p.get('name')} {p.get('desiredStatus')} {hours:.2f}h "
                         f"${usd:.2f} {util_txt}{'' if rate else ' rate=?'}{' grace' if in_grace else ''}")

        total = running_usd + frozen_usd + sum(self.ended.values())
        if self.max_usd and total >= self.max_usd:
            for p in running:
                grace = parse_time(((state.get("pods") or {}).get(p.get("id")) or {}).get("grace_until"))
                if grace is None or now >= grace:
                    reasons.setdefault(p.get("id"), f"max-usd ${self.max_usd:g} reached (est ${total:.2f})")
        running_ids = {p.get("id") for p in running}
        for pid, reason in self.pending.items():
            if pid in running_ids:
                reasons.setdefault(pid, reason + " [retry]")
        by_id = {p.get("id"): p for p in running}
        actions = []
        for pid, reason in sorted(reasons.items()):
            rec = (state.get("pods") or {}).get(pid, {})
            actions.append((pid, "terminate" if has_network_volume(by_id[pid], rec) else "stop", reason))
        line = (f"{iso(now)} tribe-pods={len(listed)} est=${total:.2f}/${self.max_usd:g} "
                f"burn=${rate_total:.2f}/h" + (" | " + " | ".join(parts) if parts else ""))
        return line, actions


def fetch_telemetry() -> tuple[dict | None, str]:
    try:
        me = graphql(Q_TELEMETRY).get("myself") or {}
    except ApiError as e:
        return None, f" telemetry unavailable ({e})"
    tel = {p.get("id"): p.get("runtime") for p in me.get("pods") or []}
    extra = (f" balance=${to_float(me.get('clientBalance')):.2f} "
             f"account-burn=${to_float(me.get('currentSpendPerHr')):.2f}/h")
    return tel, extra


def cmd_watchdog(args) -> int:
    if args.max_usd <= 0 or args.max_hours <= 0:
        raise UsageError("--max-usd and --max-hours must be > 0")
    need("RUNPOD_API_KEY")  # fail now, not on every tick
    wd = Watchdog(args.max_usd, args.max_hours, args.idle_minutes)
    fails = 0
    out(f"watchdog start: caps ${args.max_usd:g} total, {args.max_hours:g} h per pod, "
        f"idle {args.idle_minutes:g} min, every {args.interval}s, prefix {PREFIX!r}")
    while True:
        try:
            now = now_ts()
            try:
                pods = rest("GET", "/pods") or []
            except ApiError as e:
                fails += 1
                out(f"{iso(now)} ERROR listing pods ({fails} in a row), no decision this tick: {e}")
                pods = None
            if pods is not None:
                fails = 0
                telemetry, extra = fetch_telemetry()
                line, actions = wd.tick(pods, telemetry, load_state(), now)
                out(line + extra)
                for pid, action, reason in actions:
                    try:
                        (terminate_pod if action == "terminate" else stop_pod)(pid, now)
                        wd.pending.pop(pid, None)
                        note = "" if action == "terminate" else " (pod volume kept: pull, then pod-terminate --yes)"
                        out(f"{iso(now)} {'TERMINATED' if action == 'terminate' else 'STOPPED'} {pid}: {reason}{note}")
                    except ApiError as e:
                        wd.pending[pid] = reason
                        out(f"{iso(now)} {action.upper()} FAILED {pid} ({reason}), retrying next tick: {e}")
                write_json_atomic(HEARTBEAT_FILE, {
                    "epoch": now, "ts": iso(now), "pid": os.getpid(), "interval": args.interval,
                    "max_usd": args.max_usd, "max_hours": args.max_hours,
                })
        except KeyboardInterrupt:
            raise
        except Exception as e:  # the loop must survive anything
            out(f"{iso(now_ts())} ERROR in watchdog tick: {type(e).__name__}: {e}")
        if args.once:
            return 0
        _sleep(args.interval)


# --------------------------------------------------------------------------- cli

_sleep = time.sleep


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("gpus", help="GPU types by price: community/secure, on-demand/spot, stock, VRAM, host RAM")
    s.add_argument("--dc", help="datacenter id, e.g. EU-RO-1")
    s.add_argument("--gpu-count", type=int, default=1, help="GPUs per pod")
    s.add_argument("--min-ram-gb", type=int, default=48, help="host RAM per GPU (default 48)")
    s.add_argument("--min-vram-gb", type=int, default=24, help="hide smaller GPUs (default 24)")
    s.add_argument("--all", action="store_true", help="include every type, in stock or not")
    s.set_defaults(fn=cmd_gpus)

    s = sub.add_parser("volume-create", help="create a network volume (optional; Secure Cloud only)")
    s.add_argument("--name", required=True)
    s.add_argument("--size-gb", type=int, required=True)
    s.add_argument("--dc", required=True)
    s.set_defaults(fn=cmd_volume_create)

    s = sub.add_parser("volume-list", help="list network volumes")
    s.set_defaults(fn=cmd_volume_list)

    s = sub.add_parser("pod-create", help="create a pod with full SSH and /workspace (pod or network volume)")
    s.add_argument("--name", required=True, help=f"must start with {PREFIX!r}")
    s.add_argument("--gpu-type", required=True, help="e.g. 4090, 3090, A40, A6000, L40S, 'H100 80GB HBM3'")
    s.add_argument("--gpu-count", type=int, default=1)
    s.add_argument("--cloud", default="COMMUNITY", choices=["COMMUNITY", "SECURE"])
    s.add_argument("--interruptible", action="store_true",
                   help="spot pod: cheaper, but RunPod may stop it at any time (default on-demand)")
    s.add_argument("--volume-id", help="network volume (Secure Cloud). Default: a pod volume instead")
    s.add_argument("--volume-gb", type=int, default=120,
                   help="pod volume at /workspace when no --volume-id (deleted on terminate)")
    s.add_argument("--container-disk-gb", type=int, default=30)
    s.add_argument("--min-ram-gb", type=int, default=48, help="host RAM per GPU (REST minRAMPerGPU)")
    s.add_argument("--min-cuda", default=DEFAULT_MIN_CUDA, choices=CUDA_VERSIONS,
                   help="lowest host driver CUDA version (whisperx needs torch 2.8 / CUDA 12.8)")
    s.add_argument("--image", default=DEFAULT_IMAGE)
    s.add_argument("--max-hours", type=float, required=True, help="watchdog stops/terminates the pod after this")
    s.add_argument("--dc", help="datacenter; with --volume-id it is taken from the volume")
    s.add_argument("--public-key-file", default=str(PUBKEY_FILE))
    s.add_argument("--dry-run", action="store_true", help="print the (redacted) request, send nothing")
    s.add_argument("--allow-no-watchdog", action="store_true")
    s.set_defaults(fn=cmd_pod_create)

    s = sub.add_parser("pod-wait", help="wait for SSH; prints `export POD=... POD_PORT=...`")
    s.add_argument("pod_id")
    s.add_argument("--timeout-min", type=float, default=30)
    s.add_argument("--interval", type=float, default=10)
    s.set_defaults(fn=cmd_pod_wait)

    s = sub.add_parser("pod-list", help="list pods")
    s.add_argument("--json", action="store_true", help="full JSON, env redacted")
    s.set_defaults(fn=cmd_pod_list)

    s = sub.add_parser("pod-stop", help="stop a pod: GPU billing ends, /workspace is kept")
    s.add_argument("pod_id")
    s.set_defaults(fn=cmd_pod_stop)

    s = sub.add_parser("pod-start", help="restart a stopped pod (e.g. to pull results)")
    s.add_argument("pod_id")
    s.add_argument("--grace-min", type=float, default=30,
                   help="minutes the watchdog leaves it alone (still counted in spend)")
    s.set_defaults(fn=cmd_pod_start)

    s = sub.add_parser("pod-terminate", help="delete a pod; a POD volume is deleted with it")
    s.add_argument("pod_id")
    s.add_argument("--yes", action="store_true", help="confirm deleting a pod volume")
    s.set_defaults(fn=cmd_pod_terminate)

    s = sub.add_parser("balance", help="credit balance and current spend rate")
    s.set_defaults(fn=cmd_balance)

    s = sub.add_parser("watchdog", help=f"enforce spend/time caps on {PREFIX}* pods (run under nohup)")
    s.add_argument("--max-usd", type=float, required=True, help="act on all tribe- pods at this estimated total")
    s.add_argument("--max-hours", type=float, required=True, help="per-pod ceiling (the lower of this and the pod's own)")
    s.add_argument("--interval", type=int, default=60)
    s.add_argument("--idle-minutes", type=float, default=20,
                   help="act after this long at 0%% GPU, once the pod has used the GPU (0 = off)")
    s.add_argument("--once", action="store_true", help="one tick, then exit")
    s.set_defaults(fn=cmd_watchdog)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_env()
        return args.fn(args)
    except UsageError as e:
        err(f"error: {e}")
        return 2
    except ApiError as e:
        err(f"API error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

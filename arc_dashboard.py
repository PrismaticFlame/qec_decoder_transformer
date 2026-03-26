#!/usr/bin/env python3
"""
arc_dashboard.py — Browser-based dashboard for monitoring ARC training jobs.

SETUP (each laptop session):
──────────────────────────────────────────────────────────────────────────────
  1. On ARC login node, start the server:
       python ~/arc_dashboard.py                          # view all jobs on account
       python ~/arc_dashboard.py --jobs 4828257           # view specific job
       python ~/arc_dashboard.py --jobs 4828257 4829000   # multiple jobs

  2. On your laptop, open a terminal and run (leave it minimized):
       ssh -L 5000:localhost:5000 -N USER@tinkercliffs2.arc.vt.edu

  3. Open browser:  http://localhost:5000

  Dashboard auto-refreshes every 15s. pkill -f arc_dashboard.py to stop on ARC.
──────────────────────────────────────────────────────────────────────────────
No extra Python dependencies — stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

# ── Defaults ───────────────────────────────────────────────────────────────────
PORT             = 5000
REFRESH_INTERVAL = 15   # seconds between server-side data fetches
DEFAULT_LOG_DIR  = "/home/{user}/src/trans7_alphaqubit_ckpt/logs"


# ── Parsers (shared with arc_monitor.py logic) ─────────────────────────────────
_TQDM_RE = re.compile(
    r"(\d+)/(\d+)\s+\[(\S+)<(\S+),\s*([\d.]+)it/s.*?"
    r"loss=([\d.]+).*?lr=(\S+?)(?:,|\]).*?bs=(\d+)"
)
_BEST_LER_RE = re.compile(r"Best LER:\s*([\d.]+)\s+at step\s+(\d+)")
_GPU_RE = re.compile(
    r"[\d/]+ [\d:.]+,\s*(\d+)\s*%,\s*\d+\s*%,\s*([\d.]+)\s*MiB,"
    r"\s*([\d.]+)\s*MiB,\s*(\d+),\s*([\d.]+)\s*W"
)


def _tail(path: str, n: int = 300) -> list[str]:
    try:
        with open(path, errors="replace") as f:
            return f.readlines()[-n:]
    except FileNotFoundError:
        return []


def _find_logs(log_dir: str, job_id: str) -> tuple[str, str, str]:
    """Return (out_log, err_log, gpu_log) paths for a job ID."""
    try:
        files = [
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if job_id in f
        ]
    except FileNotFoundError:
        files = []
    out = next((f for f in files if f.endswith(".out")), f"{log_dir}/trans7_ddp_{job_id}.out")
    err = next((f for f in files if f.endswith(".err")), f"{log_dir}/trans7_ddp_{job_id}.err")
    gpu = next((f for f in files if "gpu_usage" in f),   f"{log_dir}/gpu_usage_{job_id}.log")
    return out, err, gpu


def _parse_tqdm(lines):
    for line in reversed(lines):
        m = _TQDM_RE.search(line)
        if m:
            step, total, elapsed, eta, speed, loss, lr, bs = m.groups()
            return {
                "step": int(step), "total": int(total),
                "elapsed": elapsed, "eta": eta,
                "speed": float(speed), "loss": float(loss),
                "lr": lr, "bs": int(bs),
            }
    return None


def _parse_best_ler(lines):
    result = None
    for line in lines:
        m = _BEST_LER_RE.search(line)
        if m:
            result = {"ler": float(m.group(1)), "step": int(m.group(2))}
    return result


def _parse_gpus(lines):
    readings = []
    for line in lines:
        m = _GPU_RE.search(line)
        if m:
            readings.append({
                "util": int(m.group(1)),
                "mem_used_mib": float(m.group(2)),
                "mem_total_mib": float(m.group(3)),
                "temp": int(m.group(4)),
                "power_w": float(m.group(5)),
            })
    return [readings[-2], readings[-1]] if len(readings) >= 2 else readings


def _parse_errors(lines):
    skip = ("it/s", "██", "░", "site-packages", "return Variable",
            "UserWarning", "Triggered internally", "OMP_NUM_THREADS",
            "***", "Setting OMP", "W0", "libyaml")
    result = []
    for line in lines:
        s = line.strip()
        if not s or any(t in s for t in skip):
            continue
        if re.match(r"pretrain_\S+:\s+\d+%\|", s):
            continue
        result.append(s)
    return result[-10:]


def discover_jobs(account: str | None, users: list[str]) -> dict[str, str]:
    """
    Query squeue and return {job_id: username} for all matching jobs.
    Filters by account and/or users; both are optional.
    """
    cmd = ["squeue", "--format=%i %u", "--noheader"]
    if account:
        cmd += ["-A", account]
    if users:
        cmd += ["-u", ",".join(users)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        result = {}
        for line in out.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                result[parts[0]] = parts[1]
        return result
    except Exception:
        return {}


def fetch_job_data(job_id: str, log_dir_template: str, username: str = "") -> dict:
    """Collect all data for one job. Called from the background refresh thread."""
    owner   = username or os.environ["USER"]
    log_dir = log_dir_template.format(user=owner)
    out_log, err_log, gpu_log = _find_logs(log_dir, job_id)

    out_lines = _tail(out_log)
    err_lines = _tail(err_log)
    gpu_lines = _tail(gpu_log, n=30)

    # squeue
    try:
        sq = subprocess.run(
            ["squeue", "-j", job_id, "--format=%i %T %M %L", "--noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        parts = sq.split()
        slurm = {
            "state":      parts[1] if len(parts) > 1 else "?",
            "elapsed":    parts[2] if len(parts) > 2 else "?",
            "remaining":  parts[3] if len(parts) > 3 else "?",
            "start_time": None,
        } if parts else None
        # For pending jobs, fetch estimated start time separately
        if slurm and slurm["state"] == "PENDING":
            sq2 = subprocess.run(
                ["squeue", "-j", job_id, "--start", "--format=%S", "--noheader"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if sq2:
                slurm["start_time"] = sq2
    except Exception:
        slurm = None

    tqdm     = _parse_tqdm(err_lines)
    best_ler = _parse_best_ler(out_lines)
    gpus     = _parse_gpus(gpu_lines)
    errors   = _parse_errors(err_lines)

    eta_h = None
    if tqdm and tqdm["speed"] > 0:
        eta_h = round((tqdm["total"] - tqdm["step"]) / tqdm["speed"] / 3600, 1)

    return {
        "job_id":   job_id,
        "username": username,
        "slurm":    slurm,
        "tqdm":     tqdm,
        "eta_h":    eta_h,
        "best_ler": best_ler,
        "gpus":     gpus,
        "errors":   errors,
        "log_dir":  log_dir,
        "updated":  datetime.now().strftime("%H:%M:%S"),
    }


# ── Background refresh thread ──────────────────────────────────────────────────
_cache: dict = {}
_cache_lock = threading.Lock()


def _refresh_loop(account: str | None, users: list[str], log_dir_template: str,
                  static_jobs: dict[str, str]):
    """
    static_jobs: {job_id: username} for jobs passed via --jobs
    account/users: used to dynamically discover jobs from squeue each cycle
    """
    while True:
        live_jobs: dict[str, str] = dict(static_jobs)
        if account or users:
            live_jobs.update(discover_jobs(account, users))

        # Evict jobs no longer in squeue (static --jobs entries are always kept)
        with _cache_lock:
            for jid in [jid for jid in _cache if jid not in live_jobs]:
                del _cache[jid]

        for jid, username in live_jobs.items():
            data = fetch_job_data(jid, log_dir_template, username)
            with _cache_lock:
                _cache[jid] = data

        time.sleep(REFRESH_INTERVAL)


# ── HTTP handler ───────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, *_):
        pass  # silence request logs

    def do_GET(self):
        if self.path == "/api/status":
            with _cache_lock:
                payload = {
                    "jobs":             dict(_cache),
                    "server_time":      datetime.now().strftime("%H:%M:%S"),
                    "refresh_interval": REFRESH_INTERVAL,
                }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path in ("/", "/index.html"):
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()


# ── HTML + JS (embedded) ───────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARC Dashboard</title>
<style>
  :root {
    --bg:      #1e1e2e;
    --surface: #2a2a3e;
    --border:  #44475a;
    --text:    #cdd6f4;
    --dim:     #6c7086;
    --cyan:    #89dceb;
    --green:   #a6e3a1;
    --yellow:  #f9e2af;
    --red:     #f38ba8;
    --blue:    #89b4fa;
    --mauve:   #cba6f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 13px; }
  header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 20px;
           display: flex; align-items: center; justify-content: space-between; }
  header h1 { color: var(--cyan); font-size: 16px; letter-spacing: 1px; }
  #meta { color: var(--dim); font-size: 12px; }
  #meta span { color: var(--text); }
  #countdown { color: var(--yellow); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
          gap: 16px; padding: 16px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
          padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .card-header { display: flex; align-items: center; justify-content: space-between; }
  .job-id { color: var(--cyan); font-size: 15px; font-weight: bold; }
  .username { color: var(--dim); font-size: 12px; }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
  .badge-running  { background: #1c4a2e; color: var(--green); }
  .badge-pending  { background: #3d3005; color: var(--yellow); }
  .badge-finished { background: #3a1a2a; color: var(--red); }
  .badge-unknown  { background: #2a2a3e; color: var(--dim); }
  .times { color: var(--dim); font-size: 11px; }
  .times span { color: var(--text); }
  section { display: flex; flex-direction: column; gap: 6px; }
  .section-title { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  .progress-track { background: var(--border); border-radius: 3px; height: 8px; width: 100%; overflow: hidden; }
  .progress-fill  { height: 100%; border-radius: 3px; background: var(--blue); transition: width 0.5s; }
  .progress-label { display: flex; justify-content: space-between; color: var(--dim); font-size: 11px; }
  .progress-label .pct { color: var(--blue); font-weight: bold; }
  .stats { display: flex; gap: 20px; flex-wrap: wrap; }
  .stat label { color: var(--dim); font-size: 10px; display: block; text-transform: uppercase; }
  .stat value { color: var(--text); font-size: 13px; }
  .stat value.loss { color: var(--yellow); }
  .stat value.ler  { color: var(--green); font-weight: bold; }
  .stat value.eta  { color: var(--mauve); }
  .gpu-row { display: flex; align-items: center; gap: 8px; }
  .gpu-label { color: var(--dim); width: 42px; flex-shrink: 0; }
  .gpu-track { flex: 1; background: var(--border); border-radius: 3px; height: 14px; overflow: hidden; position: relative; }
  .gpu-fill   { height: 100%; border-radius: 3px; transition: width 0.5s; }
  .fill-high  { background: #1c4a2e; }
  .fill-mid   { background: #3d3005; }
  .fill-low   { background: #4a1c2e; }
  .gpu-pct    { position: absolute; right: 6px; top: 0; line-height: 14px; font-size: 11px; color: var(--text); }
  .gpu-meta   { color: var(--dim); font-size: 11px; width: 220px; flex-shrink: 0; text-align: right; }
  .no-gpu     { color: var(--dim); font-style: italic; }
  .errors     { color: var(--red); font-size: 11px; white-space: pre-wrap; word-break: break-all; }
  .no-errors  { color: var(--dim); font-style: italic; }
  .updated    { color: var(--dim); font-size: 10px; text-align: right; }
  footer { text-align: center; padding: 16px; color: var(--dim); font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>⬡ ARC Monitor</h1>
  <div id="meta">
    server time: <span id="serverTime">—</span>
    &nbsp;|&nbsp;
    next refresh: <span id="countdown" class="countdown">—</span>
  </div>
</header>
<div class="grid" id="grid">
  <div class="card"><div class="section-title">Loading...</div></div>
</div>
<footer>ARC Dashboard — running on login node — data from SLURM + log files</footer>

<script>
let refreshInterval = 15;
let countdown = refreshInterval;
let timer;

function humanRemaining(s) {
  if (!s || s === '?') return '';
  let days = 0, hours = 0, mins = 0;
  const dash = s.indexOf('-');
  if (dash !== -1) { days = parseInt(s); s = s.substring(dash + 1); }
  const p = s.split(':');
  if (p.length >= 3) {
    // H:MM:SS
    hours = parseInt(p[0]); mins = parseInt(p[1]);
  } else if (p.length === 2) {
    // MM:SS (under 1 hour) — first segment is minutes, not hours
    mins = parseInt(p[0]);
  }
  const totalH = days * 24 + hours + Math.round(mins / 60);
  if (days > 0)  return `~${totalH}h (${days}d ${hours}h)`;
  if (totalH > 0) return `~${totalH}h`;
  return `~${mins}m`;
}

function badge(state) {
  if (!state) return '<span class="badge badge-unknown">UNKNOWN</span>';
  const cls = state === 'RUNNING' ? 'running' : state === 'PENDING' ? 'pending' :
              state === 'COMPLETED' ? 'finished' : 'unknown';
  return `<span class="badge badge-${cls}">${state}</span>`;
}

function gpuFillClass(util) {
  return util >= 80 ? 'fill-high' : util >= 40 ? 'fill-mid' : 'fill-low';
}

function renderJob(d) {
  const slurm    = d.slurm || {};
  const tqdm     = d.tqdm  || null;
  const best_ler = d.best_ler || null;
  const gpus     = d.gpus  || [];
  const errors   = d.errors || [];

  const pct = tqdm ? (100 * tqdm.step / tqdm.total).toFixed(2) : null;

  // SLURM
  let slurmHtml;
  if (!slurm.state) {
    slurmHtml = `<div class="times" style="color:var(--red)">Job not found — finished or not started</div>`;
  } else if (slurm.state === 'PENDING') {
    const startStr = (slurm.start_time && slurm.start_time !== 'N/A')
      ? slurm.start_time.replace('T', ' ')
      : 'unknown';
    slurmHtml = `<div class="times">Est. start: <span style="color:var(--yellow)">${startStr}</span></div>`;
  } else {
    slurmHtml = `<div class="times">Elapsed: <span>${slurm.elapsed}</span> <span style="color:var(--dim)">${humanRemaining(slurm.elapsed)}</span>  &nbsp;  Remaining: <span>${slurm.remaining}</span> <span style="color:var(--dim)">${humanRemaining(slurm.remaining)}</span></div>`;
  }

  // Training progress
  let trainingHtml = '<div class="no-gpu">Waiting for training output...</div>';
  if (tqdm) {
    const etaStr = d.eta_h != null ? `~${d.eta_h}h` : tqdm.eta;
    trainingHtml = `
      <div class="progress-label">
        <span><b>${tqdm.step.toLocaleString()}</b> / ${tqdm.total.toLocaleString()} steps</span>
        <span class="pct">${pct}%</span>
      </div>
      <div class="progress-track"><div class="progress-fill" style="width:${pct}%"></div></div>
      <div class="stats">
        <div class="stat"><label>Loss</label><value class="loss">${tqdm.loss.toFixed(4)}</value></div>
        <div class="stat"><label>LR</label><value>${tqdm.lr}</value></div>
        <div class="stat"><label>Batch</label><value>${tqdm.bs}</value></div>
        <div class="stat"><label>Speed</label><value>${tqdm.speed.toFixed(2)} it/s</value></div>
        <div class="stat"><label>ETA</label><value class="eta">${etaStr}</value></div>
        ${best_ler ? `<div class="stat"><label>Best LER</label><value class="ler">${best_ler.ler.toFixed(6)}</value></div>` : ''}
        ${best_ler ? `<div class="stat"><label>Best Step</label><value>${best_ler.step.toLocaleString()}</value></div>` : ''}
      </div>`;
  }

  // GPU
  let gpuHtml = '<div class="no-gpu">No GPU data — add nvidia-smi monitor to SLURM script</div>';
  if (gpus.length > 0) {
    gpuHtml = gpus.map((g, i) => {
      const memGb    = (g.mem_used_mib / 1024).toFixed(1);
      const totGb    = (g.mem_total_mib / 1024).toFixed(0);
      const memPct   = (100 * g.mem_used_mib / g.mem_total_mib).toFixed(0);
      return `
        <div class="gpu-row">
          <span class="gpu-label">GPU ${i}</span>
          <div class="gpu-track">
            <div class="gpu-fill ${gpuFillClass(g.util)}" style="width:${g.util}%"></div>
            <span class="gpu-pct">${g.util}%</span>
          </div>
          <span class="gpu-meta">${memGb}/${totGb} GB (${memPct}%) · ${g.temp}°C · ${g.power_w.toFixed(0)}W</span>
        </div>`;
    }).join('');
  }

  // Errors
  const errHtml = errors.length
    ? `<pre class="errors">${errors.join('\\n')}</pre>`
    : `<span class="no-errors">None</span>`;

  return `
    <div class="card">
      <div class="card-header">
        <div style="display:flex;align-items:center;gap:8px">
          <span class="job-id">Job ${d.job_id}</span>
          ${d.username ? `<span class="username">@${d.username}</span>` : ''}
        </div>
        ${badge(slurm.state)}
      </div>
      ${slurmHtml}

      <section>
        <div class="section-title">Training</div>
        ${trainingHtml}
      </section>

      <section>
        <div class="section-title">GPU</div>
        ${gpuHtml}
      </section>

      <section>
        <div class="section-title">Errors</div>
        ${errHtml}
      </section>

      <div class="updated">updated ${d.updated}</div>
    </div>`;
}

function render(data) {
  document.getElementById('serverTime').textContent = data.server_time;
  refreshInterval = data.refresh_interval;
  const grid = document.getElementById('grid');
  const jobs = Object.values(data.jobs);
  if (jobs.length === 0) {
    grid.innerHTML = '<div class="card"><div class="section-title">No job data yet...</div></div>';
    return;
  }
  grid.innerHTML = jobs.map(renderJob).join('');
}

function fetchStatus() {
  fetch('/api/status')
    .then(r => r.json())
    .then(data => { render(data); countdown = refreshInterval; })
    .catch(() => { /* keep showing last data */ });
}

function tick() {
  countdown--;
  document.getElementById('countdown').textContent = countdown + 's';
  if (countdown <= 0) fetchStatus();
}

fetchStatus();
setInterval(tick, 1000);
</script>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    global REFRESH_INTERVAL

    parser = argparse.ArgumentParser(description="ARC web dashboard")
    parser.add_argument("--jobs",     nargs="*", default=[],  metavar="JOB_ID",
                        help="Explicit SLURM job ID(s) to monitor")
    parser.add_argument("--account",  default="quantum_training",
                        help="SLURM account name — monitor all jobs under this account")
    parser.add_argument("--users",    nargs="*", default=['cdw24', 'tzuchen', 'arjuns8'],  metavar="USER",
                        help="Username(s) — monitor all jobs submitted by these users")
    parser.add_argument("--log-dir",  default=None,
                        help="Log directory override (default: trans7_alphaqubit_ckpt/logs)")
    parser.add_argument("--port",     type=int, default=PORT, help=f"Port (default: {PORT})")
    parser.add_argument("--interval", type=int, default=REFRESH_INTERVAL,
                        help=f"Refresh interval in seconds (default: {REFRESH_INTERVAL})")
    args = parser.parse_args()

    # if not args.jobs and not args.account and not args.users:
    #     parser.error("Provide at least one of --jobs, --account, or --users")

    REFRESH_INTERVAL = args.interval

    user             = os.environ["USER"]
    log_dir_template = args.log_dir or DEFAULT_LOG_DIR  # keep {user} unresolved

    # Explicit jobs have no username (unknown without a squeue lookup)
    static_jobs: dict[str, str] = {jid: "" for jid in args.jobs}

    # Prime the cache immediately before the server starts
    initial_jobs: dict[str, str] = dict(static_jobs)
    if args.account or args.users:
        initial_jobs.update(discover_jobs(args.account, args.users))

    print(f"Fetching initial data for {len(initial_jobs)} job(s)...")
    for jid, uname in initial_jobs.items():
        _cache[jid] = fetch_job_data(jid, log_dir_template, uname)
        state    = (_cache[jid]["slurm"] or {}).get("state", "not found")
        resolved = log_dir_template.format(user=uname or user)
        print(f"  {jid} ({uname or '?'}): {state}  logs: {resolved}")

    # Start background refresh thread
    t = threading.Thread(
        target=_refresh_loop,
        args=(args.account, args.users, log_dir_template, static_jobs),
        daemon=True,
    )
    t.start()

    # Start HTTP server
    server = HTTPServer(("localhost", args.port), Handler)
    print(f"\nDashboard running at http://localhost:{args.port}")
    print(f"Refreshing every {REFRESH_INTERVAL}s\n")
    print("On your laptop, run:")
    print(f"  ssh -L {args.port}:localhost:{args.port} -N {user}@$HOSTNAME")
    print(f"Then open: http://localhost:{args.port}")
    print("\nCtrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
arc_dashboard.py — Browser-based dashboard for monitoring ARC training jobs.

SETUP (each laptop session):
──────────────────────────────────────────────────────────────────────────────
  1. On ARC login node, start the server:
       python ~/arc_dashboard.py --jobs 4828257
       python ~/arc_dashboard.py --jobs 4828257 4829000   # multiple jobs

  2. On your laptop, open a terminal and run (leave it minimized):
       ssh -L 5000:localhost:5000 -N cdw24@tinkercliffs2.arc.vt.edu

  3. Open browser:  http://localhost:5000

  Dashboard auto-refreshes every 15s. Ctrl+C on ARC to stop the server.
──────────────────────────────────────────────────────────────────────────────
No extra Python dependencies — stdlib only.
"""

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


def fetch_job_data(job_id: str, log_dir: str) -> dict:
    """Collect all data for one job. Called from the background refresh thread."""
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


def _refresh_loop(job_ids: list[str], log_dirs: dict[str, str]):
    while True:
        for jid in job_ids:
            data = fetch_job_data(jid, log_dirs[jid])
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
  if (p.length >= 2) { hours = parseInt(p[0]); mins = parseInt(p[1]); }
  const totalH = days * 24 + hours + Math.round(mins / 60);
  return days > 0 ? `~${totalH}h (${days}d ${hours}h)` : `~${totalH}h`;
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
        <span class="job-id">Job ${d.job_id}</span>
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
    parser.add_argument("--jobs",     nargs="+", required=True, help="SLURM job ID(s)")
    parser.add_argument("--log-dir",  default=None,
                        help="Log directory override (default: trans7_alphaqubit_ckpt/logs)")
    parser.add_argument("--port",     type=int, default=PORT, help=f"Port (default: {PORT})")
    parser.add_argument("--interval", type=int, default=REFRESH_INTERVAL,
                        help=f"Refresh interval in seconds (default: {REFRESH_INTERVAL})")
    args = parser.parse_args()

    REFRESH_INTERVAL = args.interval

    user = os.environ.get("USER", "cdw24")
    default_log_dir = DEFAULT_LOG_DIR.format(user=user)

    # Build per-job log dir mapping
    log_dirs = {jid: (args.log_dir or default_log_dir) for jid in args.jobs}

    # Prime the cache immediately before the server starts
    print(f"Fetching initial data for job(s): {', '.join(args.jobs)}")
    for jid in args.jobs:
        _cache[jid] = fetch_job_data(jid, log_dirs[jid])
        state = (_cache[jid]["slurm"] or {}).get("state", "not found")
        print(f"  {jid}: {state}  logs: {log_dirs[jid]}")

    # Start background refresh thread
    t = threading.Thread(target=_refresh_loop, args=(args.jobs, log_dirs), daemon=True)
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
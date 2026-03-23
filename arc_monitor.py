#!/usr/bin/env python3
"""
arc_monitor.py — Live dashboard for monitoring Trans7 training on ARC HPC.

TWO MODES:

  1. LOCAL (run on your laptop, SSH to ARC):
       python arc_monitor.py --job 4838744
     ARC uses Duo 2FA, so SSH multiplexing is used to authenticate once
     and reuse that connection for every refresh.  When you run the script
     it will open ONE interactive SSH session — complete the Duo push in
     your terminal — then the dashboard starts and all subsequent fetches
     are silent.

  2. REMOTE (run directly on the ARC login node — no SSH needed):
       python arc_monitor.py --job 4838744 --local
     SSH into ARC first, then run the script there.  It reads log files
     directly from disk.  Rich must be installed in your conda env or
     base env on ARC:  pip install rich

QUICK DEBUG (prints raw SSH output, skips the dashboard):
       python arc_monitor.py --job 4838744 --debug

Requirements (local machine):
    pip install rich
"""

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_HOST    = "tinkercliffs2.arc.vt.edu"
DEFAULT_USER    = "cdw24"
DEFAULT_LOG_DIR = "/home/{user}/src/trans7_alphaqubit_ckpt/logs"
DEFAULT_INTERVAL = 15  # seconds

# ── SSH / local fetch ──────────────────────────────────────────────────────────
# Windows OpenSSH does not support ControlMaster/ControlPath (Unix sockets).
# Instead we do one plain SSH call at startup so Duo fires once interactively,
# then rely on Duo's session memory for the silent refresh calls.
# If Duo prompts on every call (uncommon), add ControlMaster to ~/.ssh/config
# manually — the script will benefit from it automatically.

def _ssh_base(user: str, host: str) -> list[str]:
    return [
        "ssh",
        "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no",
        f"{user}@{host}",
    ]


def open_tunnel(user: str, host: str) -> bool:
    """
    Run one interactive SSH command so Duo/2FA fires now, before the dashboard
    takes over the terminal.  After this succeeds, Duo's session memory means
    the silent refresh calls won't re-prompt (typically valid for hours).
    """
    console = Console()
    console.print(f"\n[cyan]Connecting to {user}@{host}[/cyan]")
    console.print("[dim]Approve the Duo push below, then the dashboard will start.[/dim]\n")
    try:
        r = subprocess.run(
            _ssh_base(user, host) + ["echo connected"],
            timeout=60,
            # stdin/stdout/stderr inherited so Duo prompt shows in terminal
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        console.print("[red]Timed out waiting for SSH / Duo.[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]'ssh' not found. Is OpenSSH installed and on PATH?[/red]")
        sys.exit(1)


def ssh(user: str, host: str, cmd: str, timeout: int = 20) -> tuple[str, str]:
    """Run cmd on remote host. Returns (stdout, stderr)."""
    try:
        r = subprocess.run(
            _ssh_base(user, host) + [cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return "", "SSH command timed out"
    except Exception as e:
        return "", str(e)


def read_local(path: str, n_lines: int = 300) -> str:
    """Read last n_lines of a local file (for --local mode on ARC)."""
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except FileNotFoundError:
        return ""


def fetch_sections_ssh(user: str, host: str, out_log: str, err_log: str,
                        gpu_log: str, job_id: str) -> dict[str, list[str]]:
    """Fetch all log data in a single SSH round-trip."""
    batch_cmd = (
        f"echo '===OUT==='; tail -n 300 {out_log} 2>/dev/null; "
        f"echo '===ERR==='; tail -n 300 {err_log} 2>/dev/null; "
        f"echo '===GPU==='; tail -n 30  {gpu_log} 2>/dev/null; "
        f"echo '===SQ===';  squeue -j {job_id} --format='%T %M %L' --noheader 2>/dev/null"
    )
    raw, err = ssh(user, host, batch_cmd)
    if not raw and err:
        # Return error hint in a special key
        return {"OUT": [], "ERR": [], "GPU": [], "SQ": [], "_ssh_err": [err.strip()]}
    return _split_sections(raw)


def fetch_sections_local(out_log: str, err_log: str, gpu_log: str,
                          job_id: str) -> dict[str, list[str]]:
    """Read log data directly from disk (--local mode)."""
    out  = read_local(out_log)
    err  = read_local(err_log)
    gpu  = read_local(gpu_log, n_lines=30)
    try:
        sq = subprocess.run(
            ["squeue", "-j", job_id, "--format=%T %M %L", "--noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        sq = ""
    raw = f"===OUT===\n{out}===ERR===\n{err}===GPU===\n{gpu}===SQ===\n{sq}"
    return _split_sections(raw)


def _split_sections(raw: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {"OUT": [], "ERR": [], "GPU": [], "SQ": [], "_ssh_err": []}
    current = None
    for line in raw.splitlines():
        if line in ("===OUT===", "===ERR===", "===GPU===", "===SQ==="):
            current = line[3:-3]
        elif current is not None:
            sections[current].append(line)
    return sections


# ── Parsers ────────────────────────────────────────────────────────────────────
_TQDM_RE = re.compile(
    r"(\d+)/(\d+)\s+\[(\S+)<(\S+),\s*([\d.]+)it/s.*?"
    r"loss=([\d.]+).*?lr=(\S+?)(?:,|\]).*?bs=(\d+)"
)

def parse_tqdm(lines: list[str]) -> dict | None:
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


_BEST_LER_RE = re.compile(r"Best LER:\s*([\d.]+)\s+at step\s+(\d+)")

def parse_best_ler(lines: list[str]) -> tuple[float, int] | None:
    result = None
    for line in lines:
        m = _BEST_LER_RE.search(line)
        if m:
            result = (float(m.group(1)), int(m.group(2)))
    return result


_GPU_RE = re.compile(
    r"[\d/]+ [\d:.]+,\s*(\d+)\s*%,\s*\d+\s*%,\s*([\d.]+)\s*MiB,"
    r"\s*([\d.]+)\s*MiB,\s*(\d+),\s*([\d.]+)\s*W"
)

def parse_gpu_csv(lines: list[str]) -> list[dict]:
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
    if len(readings) >= 2:
        return [readings[-2], readings[-1]]
    return readings


def parse_squeue(raw: str) -> dict | None:
    line = raw.strip()
    if not line or "Invalid job" in line:
        return None
    parts = line.split()
    return {
        "state":     parts[0] if len(parts) > 0 else "?",
        "elapsed":   parts[1] if len(parts) > 1 else "?",
        "remaining": parts[2] if len(parts) > 2 else "?",
    }


def parse_errors(lines: list[str]) -> list[str]:
    skip = ("it/s", "██", "░", "site-packages", "return Variable",
            "UserWarning", "Triggered internally", "OMP_NUM_THREADS",
            "***", "Setting OMP", "W0", "libyaml")
    errors = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if any(tok in s for tok in skip):
            continue
        if re.match(r"pretrain_\S+:\s+\d+%\|", s):
            continue
        errors.append(s)
    return errors


# ── Rendering ──────────────────────────────────────────────────────────────────
def progress_bar(pct: float, width: int = 24) -> str:
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)


def gpu_bar(util: int, width: int = 18) -> Text:
    filled = int(width * util / 100)
    bar_str = "█" * filled + "░" * (width - filled)
    color = "green" if util >= 80 else "yellow" if util >= 40 else "red"
    t = Text()
    t.append(bar_str, style=color)
    return t


def build_layout(
    job_id: str,
    job_info: dict | None,
    tqdm_info: dict | None,
    best_ler: tuple | None,
    gpus: list[dict],
    errors: list[str],
    ssh_err: str,
    last_update: str,
    interval: int,
    local_mode: bool,
) -> Layout:

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=1),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=3),
        Layout(name="right", ratio=2),
    )
    layout["left"].split_column(
        Layout(name="job_status", size=7),
        Layout(name="training"),
    )
    layout["right"].split_column(
        Layout(name="gpu"),
        Layout(name="errors"),
    )

    mode_tag = "[dim](local)[/dim]" if local_mode else "[dim](ssh)[/dim]"
    conn_status = f"[red]{ssh_err}[/red]" if ssh_err else "[green]connected[/green]"
    layout["header"].update(Panel(
        f"[bold cyan]ARC Monitor[/bold cyan]  "
        f"[dim]job[/dim] [yellow]{job_id}[/yellow]  "
        f"{mode_tag}  {conn_status}  "
        f"[dim]updated[/dim] {last_update}",
        box=box.HORIZONTALS, border_style="dim",
    ))

    # Job status
    if job_info:
        state = job_info["state"]
        sc = "green" if state == "RUNNING" else "yellow" if state == "PENDING" else "red"
        job_text = (
            f"Status:    [{sc}]{state}[/{sc}]\n"
            f"Elapsed:   [cyan]{job_info['elapsed']}[/cyan]\n"
            f"Remaining: [cyan]{job_info['remaining']}[/cyan]"
        )
    else:
        job_text = "[red]Not found[/red] — job finished or not yet started"
    layout["job_status"].update(Panel(job_text, title="[bold]SLURM Job[/bold]", border_style="blue"))

    # Training progress
    if tqdm_info:
        pct = 100 * tqdm_info["step"] / tqdm_info["total"] if tqdm_info["total"] else 0
        bar = progress_bar(pct)
        steps_left = tqdm_info["total"] - tqdm_info["step"]
        eta_h = f"{steps_left / tqdm_info['speed'] / 3600:.1f}h - {steps_left / tqdm_info['speed'] / 86400:.1f}d" if tqdm_info["speed"] > 0 else "?"

        t = Text()
        t.append(f"{tqdm_info['step']:,}", style="bold white")
        t.append(f" / {tqdm_info['total']:,} steps  ")
        t.append(f"({pct:.2f}%)\n", style="cyan")
        t.append(f"{bar}\n\n")
        t.append("Loss:  "); t.append(f"{tqdm_info['loss']:.4f}", style="yellow")
        t.append(f"    LR: {tqdm_info['lr']}    BS: {tqdm_info['bs']}\n")
        t.append(f"Speed: {tqdm_info['speed']:.2f} it/s")
        t.append(f"    ETA: ~{eta_h}\n")
        t.append(f"Elapsed: {tqdm_info['elapsed']}    tqdm ETA: {tqdm_info['eta']}")
        if best_ler:
            t.append("\n\nBest LER: ", style="dim")
            t.append(f"{best_ler[0]:.6f}", style="bold green")
            t.append(f"  at step {best_ler[1]:,}", style="dim")
    else:
        t = Text("Waiting for training output...", style="dim")
    layout["training"].update(Panel(t, title="[bold]Training[/bold]", border_style="blue"))

    # GPU
    if gpus:
        gpu_content = Text()
        for i, gpu in enumerate(gpus):
            mem_gb  = gpu["mem_used_mib"] / 1024
            tot_gb  = gpu["mem_total_mib"] / 1024
            mem_pct = 100 * gpu["mem_used_mib"] / gpu["mem_total_mib"]
            gpu_content.append(f"GPU {i}:  ")
            gpu_content.append_text(gpu_bar(gpu["util"]))
            gpu_content.append(f" {gpu['util']:3d}%\n")
            gpu_content.append(
                f"       Mem: {mem_gb:.1f}/{tot_gb:.0f} GB ({mem_pct:.0f}%)"
                f"   {gpu['temp']}°C   {gpu['power_w']:.0f} W\n\n"
            )
    else:
        gpu_content = Text("No GPU data yet", style="dim")
    layout["gpu"].update(Panel(gpu_content, title="[bold]GPU[/bold]", border_style="blue"))

    # Errors
    if errors:
        err_text = Text("\n".join(errors[-10:]), style="red")
    else:
        err_text = Text("(none)", style="dim green")
    layout["errors"].update(Panel(
        err_text, title="[bold]Errors / Warnings[/bold]",
        border_style="red" if errors else "blue",
    ))

    layout["footer"].update(
        Text(f"Refreshing every {interval}s  —  Ctrl+C to quit", style="dim", justify="center")
    )
    return layout


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live ARC training dashboard")
    parser.add_argument("--job",      required=True,               help="SLURM job ID")
    parser.add_argument("--user",     default=DEFAULT_USER,        help="ARC username")
    parser.add_argument("--host",     default=DEFAULT_HOST,        help="ARC login node hostname")
    parser.add_argument("--log-dir",  default=None,
                        help="Log directory path (on ARC). Default: trans7_alphaqubit_ckpt/logs")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Refresh interval in seconds (default: 15)")
    parser.add_argument("--local",    action="store_true",
                        help="Run directly on ARC login node — reads files locally, no SSH")
    parser.add_argument("--debug",    action="store_true",
                        help="Print raw SSH output and exit (skips dashboard, good for testing)")
    args = parser.parse_args()

    log_dir = args.log_dir or DEFAULT_LOG_DIR.format(user=args.user)
    console = Console()

    # ── Discover log file paths ──
    if args.local:
        find_out = subprocess.run(
            ["find", log_dir, "-name", f"*{args.job}*"],
            capture_output=True, text=True,
        ).stdout
    else:
        # Open the SSH tunnel interactively (handles Duo 2FA)
        if not open_tunnel(args.user, args.host):
            console.print("[red]Could not establish SSH tunnel. Exiting.[/red]")
            sys.exit(1)
        find_out, _ = ssh(args.user, args.host, f"find {log_dir} -name '*{args.job}*' 2>/dev/null")

    files    = find_out.splitlines()
    out_log  = next((f for f in files if f.endswith(".out")),      f"{log_dir}/trans7_ddp_{args.job}.out")
    err_log  = next((f for f in files if f.endswith(".err")),      f"{log_dir}/trans7_ddp_{args.job}.err")
    gpu_log  = next((f for f in files if "gpu_usage" in f),        f"{log_dir}/gpu_usage_{args.job}.log")

    console.print(f"\n[cyan]Monitoring job {args.job}[/cyan]  {'(local mode)' if args.local else f'(ssh → {args.host})'}")
    console.print(f"  out: {out_log}")
    console.print(f"  err: {err_log}")
    console.print(f"  gpu: {gpu_log}\n")

    # ── Debug mode: just dump raw data and exit ──
    if args.debug:
        console.print("[yellow]── DEBUG: raw SSH output ──[/yellow]")
        if args.local:
            sections = fetch_sections_local(out_log, err_log, gpu_log, args.job)
        else:
            sections = fetch_sections_ssh(args.user, args.host, out_log, err_log, gpu_log, args.job)
        for key in ("OUT", "ERR", "GPU", "SQ", "_ssh_err"):
            console.print(f"\n[bold cyan]=== {key} ===[/bold cyan]")
            for line in sections.get(key, []):
                console.print(line)
        return

    # ── Live dashboard ──
    time.sleep(0.5)
    with Live(console=console, refresh_per_second=0.5, screen=True) as live:
        while True:
            try:
                if args.local:
                    sections = fetch_sections_local(out_log, err_log, gpu_log, args.job)
                else:
                    sections = fetch_sections_ssh(args.user, args.host, out_log, err_log, gpu_log, args.job)

                ssh_err  = " | ".join(sections.get("_ssh_err", []))
                tqdm_info = parse_tqdm(sections["ERR"])
                best_ler  = parse_best_ler(sections["OUT"])
                gpus      = parse_gpu_csv(sections["GPU"])
                job_info  = parse_squeue("\n".join(sections["SQ"]))
                errors    = parse_errors(sections["ERR"])
                ts        = datetime.now().strftime("%H:%M:%S")

                live.update(build_layout(
                    args.job, job_info, tqdm_info, best_ler,
                    gpus, errors, ssh_err, ts, args.interval, args.local,
                ))

            except KeyboardInterrupt:
                break
            except Exception as e:
                live.update(Panel(
                    f"[red]Dashboard error: {e}[/red]\n[dim]Retrying in {args.interval}s...[/dim]"
                ))

            time.sleep(args.interval)

    console.print("[dim]Monitor stopped.[/dim]")


if __name__ == "__main__":
    main()
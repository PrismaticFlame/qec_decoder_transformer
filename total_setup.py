
# imports
import argparse
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src" / "trans7_alphaqubit"))
import move_surface_code_dirs as mscd  # type: ignore
import surface_code_dirs_finder as scd  # type: ignore


def _arc_connect_check(username: str, host: str):
    """
    REMOTE
    Confirms ARC is reachable from host, credentials work. Runs first.
    """
    print("="*60)
    print("ARC CONNECTION")
    print("="*60)
    out = subprocess.run(
        ["ssh", "-4",
         "-o", "ConnectTimeout=10",
         "-o", "ControlPath=none",
         f"{username}@{host}",
         "echo ok"],
        check=True,
        capture_output=True,
        text=True
    ).stdout
    out = out.strip()

    if (out == "ok"):
        print("Connection:", out)
        print("Connection secured, continuing.")
        return 1
    else:
        print("Connection:", out)
        print("Connection refused. See error for reason")
        return 0


def _trans7_check(username: str, host: str):
    """
    REMOTE
    Checks to ARC to see if, within the src directory, does the directory 
    'trans7_alphaqubit' exist.

    If it exists, return true

    If the directory does not exist or is not fully populated based on 
    _trans7_whitelist, return false
    """
    print("="*60)
    print("TRANS7_ALPHAQUBIT CHECK")
    print("="*60)
    out = subprocess.run(
        ["ssh", "-4",
         "-o", "ConnectTimeout=10",
         "-o", "ControlPath=none",
         f"{username}@{host}",
         "test -d ~/src/trans7_alphaqubit && echo ok || echo missing"],
         check=True,
         capture_output=True,
         text=True
    ).stdout
    out = out.strip()

    if (out == "ok"):
        print("Status:", out)
        print("Trans7 present, continuing")
        return 1
    else:
        print("Status:", out)
        print("Trans7 not present, marking for install.")
        return 0


def _trans7_whitelist(username: str, host: str):
    """
    REMOTE
    Based on trans7_whitelist list, checks if are all expected files present.

    If not all files are present, return false.

    If all files are present, return true.
    """
    trans7_files_whitelist = [".gitignore",
                              "dataset_streaming.py",
                              "dataset.py",
                              "ema.py",
                              "hyperparameters.py",
                              "layout.py",
                              "loss.py",
                              "model.py",
                              "move_surface_code_dirs.py",
                              "quick_eval_ckpt.py",
                              "README.md",
                              "run_finetune.py",
                              "run_pretrain_ddp.py",
                              "run_pretrain.py",
                              "run_test.py",
                              "run_trans7_pretrain_ddp.slurm",
                              "run_trans7_pretrain_multi_node.slurm",
                              "run_trans7_pretrain_test.slurm",
                              "surface_code_dirs_finder.py",
                              "train.py",
                              "utils.py"]
    trans7_files_present = []
    print("="*60)
    print("TRANS7 WHITELIST CHECK")
    print("="*60)
    out = subprocess.run(
        ["ssh", "-4",
         "-o", "ConnectTimeout=10",
         "-o", "ControlPath=none",
         f"{username}@{host}",
         "ls ~/src/trans7_alphaqubit 2>/dev/null || echo MISSING"],
         check=True,
         capture_output=True,
         text=True
    ).stdout
    out = out.strip()

    if out == "MISSING":
        print("Directory not found on ARC.")
        return 0
    
    trans7_files_present = out.splitlines()

    missing = [f for f in trans7_files_whitelist if f not in trans7_files_present]

    if missing:
        return missing

    print("All whitelist files present.")
    return 1
    


def _data_check(username: str, host: str):
    """
    REMOTE
    Different and not related to preparation() method.

    Checks the arc to see if '~/data/trans7_data' exists, and if so, is it populated.

    If directory is populated AND contains 'pretrain.h5', will return true.

    If directory does not exist AND/OR pretrain.h5 does not exist, return false.
    """
    print("="*60)
    print("DATA CHECK")
    print("="*60)
    out = subprocess.run(
        ["ssh", "-4",
         "-o", "ConnectTimeout=10",
         "-o", "ControlPath=none",
         f"{username}@{host}",
         "COUNT=$(find ~/src/data/trans7_data -type d -name 'surface_code_b*' 2>/dev/null | wc -l); HAS_H5=$(test -f ~/src/data/trans7_data/pretrain.h5 && echo 1 || echo 0); echo $COUNT $HAS_H5"],
         check=True,
         capture_output=True,
         text=True
    ).stdout
    parts = out.split()
    count = int(parts[0])
    has_h5 = int(parts[1])

    print("Number of data directories:", count)

    if (count == 130 and has_h5 == 1):
        print("Status:", out)
        print("Data present, continuing")
        return 1
    else:
        print(f"DATA INCOMPLETE: {count}/130 dirs, pretrain.h5={'yes' if has_h5 else 'no'}")
        return 0


def _arc_monitor_check(username: str, host: str):
    """
    REMOTE
    Check if arc_monitor.py, arc_dashboard.py, and ARC_MONITOR.md exist on ARC.

    If not, return false

    If yes, return true
    """
    print("="*60)
    print("ARC MONITOR CHECK")
    print("="*60)
    out = subprocess.run(
        ["ssh", "-4",
         "-o", "ConnectTimeout=10",
         "-o", "ControlPath=none",
         f"{username}@{host}",
         "test -f ~/src/arc_dashboard.py && test -f ~/src/arc_monitor.py && test -f ~/src/ARC_MONITOR.md && echo ok || echo missing"],
         check=True,
         capture_output=True,
         text=True
    ).stdout
    out = out.strip()

    if (out == "ok"):
        print("Status:", out)
        print("Arc Monitor present, continuing")
        return 1
    else:
        print("Status:", out)
        print("Arc Monitor not present, marking for install.")
        return 0


def _find_surface_codes(repo_root: Path) -> Path | int:
    """
    LOCAL MACHINE
    Does check 1. of preparation. 

    Finds and returns the directory that contains 130 'surface_code_b*' directories. 
    If that fails, -1 is returned.
    """
    all_dirs = scd.find_all(repo_root)
    if not all_dirs:
        return -1
    groups    = scd.group_by_parent(all_dirs)
    target    = scd.pick_target(groups)
    return target


def _move_surface_code(data_dir: Path, dest_dir: Path) -> bool:
    """
    LOCAL MACHINE
    If check 2. of preparation fails, this runs.

    Moves surface code directories into new trans7_data directory.

    Returns true if successful, returns false if unsuccessful.
    """
    return mscd.ensure_surface_code_data(data_dir, dest_dir)


def _random_sample(trans7_data: Path):
    """
    LOCAL MACHINE
    If check 4. of preparation fails, this runs.

    Randomly samples the data in the data directory to create pretrain.h5. 
    Will be extended to deal with finetuning later.
    """
    script = Path(__file__).parent / "data" / "data_random_sample.py"
    subprocess.run(
        ["python", str(script),
         "--data_dir", str(trans7_data),
         "--output", str(trans7_data / "pretrain.h5")],
         check=True
    )


def _slurm_template_force(repo_root: Path) -> list[Path]:
    """
    LOCAL MACHINE
    Checks that all .slurm files contain the required headers from template.slurm.
    Prints a warning for each non-conforming line but does not modify slurm files —
    enforcement is by visibility, not auto-patching, since each slurm intentionally
    differs in time/gres/etc.
    Returns list of slurm files that passed (all of them — for transfer).
    Calls _dos2unix_slurms() to strip DOS line endings.
    """
    slurm_dir    = repo_root / "src" / "trans7_alphaqubit"
    template     = slurm_dir / "template.slurm"

    if not template.exists():
        print("  WARNING: template.slurm not found — skipping template check.")
        return _dos2unix_slurms(repo_root)

    required_lines = [
        line.strip()
        for line in template.read_text().splitlines()
        if line.strip() and not line.startswith("#!")
    ]

    slurm_files = [p for p in slurm_dir.glob("*.slurm") if p.name != "template.slurm"]

    print("="*60)
    print("SLURM TEMPLATE CHECK")
    print("="*60)

    for slurm in slurm_files:
        contents = slurm.read_text()
        slurm_lines = [line.strip() for line in contents.splitlines()]
        missing = [req for req in required_lines if req not in slurm_lines]
        if missing:
            print(f"  WARNING: {slurm.name} is missing required headers:")
            for m in missing:
                print(f"    {m}")
        else:
            print(f"  OK: {slurm.name}")

    return _dos2unix_slurms(repo_root)


def _dos2unix_slurms(repo_root: Path) -> list[Path]:
    """
    LOCAL MACHINE
    Strip DOS line endings from .slurm files before transfer
    """
    slurm_dir = repo_root / "src" / "trans7_alphaqubit"
    slurm_files = list(slurm_dir.glob("*.slurm"))

    for path in slurm_files:
        content = path.read_bytes().replace(b'\r\n', b'\n')
        path.write_bytes(content)
        print(f"   dos2unix: {path.name}")

    return slurm_files


def _find_repo_root() -> Path | None:
    """
    LOCAL MACHINE
    Walk up from cwd until a directory containing both 'src' and 'data' is found.
    Walk back down if we hit filesystem without finding it.
    Returns that Path, or None if we fail in finding it.
    """
    # ---- sentinel walk up ----
    current = Path.cwd()
    while True:
        if (current / "REPO_ROOT.txt").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    # --- walk up ---
    current = Path.cwd()
    while True:
        if (current / "src").exists() and (current / "data").exists():
            return current
        parent = current.parent
        if parent == current:   # hit filesystem root, nothing found
            break
        current = parent
    
    # -- walk down --
    candidates = []
    for src_dir in Path.cwd().rglob("src"):
        if src_dir.is_dir() and (src_dir.parent / "data").is_dir():
            candidates.append(src_dir.parent)
        
    if not candidates:
        return None
    
    # Pick shallowest match - fewest path parts = closest to cwd
    return min(candidates, key=lambda p: len(p.parts))


def preparation():
    """
    LOCAL MACHINE
    First determines if full setup, partial setup, or no setup is required.
    0. If, from root, both 'src' and 'data' exist
    1. If surface_code_b* directories exist
    2. Checks if 'trans7_data' exists
    3. If trans7_data exists, is it populated
    4. If it is populated, does it contain 'pretrain.h5'

    If all checks pass, no setup is required. Depending on which check fails,
    go from there but after each process check to make sure following checks also fail.

    Caveat: If check 1. fails, can not proceed. User needs to download surface code 
    data from Drive or copyparty server.
    """
    print("="*60)
    print("Preparation to move files beginning")
    print("="*60)

    repo_root = _find_repo_root()
    if repo_root is None:
        print("ERROR: Could not locate repo root.")
        print("Ensure REPO_ROOT.txt exists at the repo root, or run from within the repo.")
        return 0
    print(f"Repo root: {repo_root}")

    # Check 1 - surface_code_b* directories
    data_dir = _find_surface_codes(repo_root)
    if data_dir == -1:
        print("ERROR: No surface_code_b* directories found.")
        print("Download surface code data from Drive or copyparty before continuing.")
        return 0
    
    # Check 2 - trans7_data_directory
    trans7_data = repo_root / "data" / "trans7_data"
    if not trans7_data.exists():
        print("trans7_data not found. Moving surface code...") # running _move_surface_code()
        if not _move_surface_code(data_dir, trans7_data):
            return 0
        
    # Check 3 - pretrain.h5
    if not (trans7_data / "pretrain.h5").exists():
        print("pretrain.h5 not found. Running random sample...") # running _random_sample()
        _random_sample(trans7_data)
    
    print("Local preparation complete.")
    return 1


def copy_manifest(status: dict, repo_root: Path) -> dict:
    """
    LOCAL MACHINE
    Calls other check methods to determine what will be copied over to the ARC
    """
    to_copy = {}

    if status["update"]:
        to_copy["update"] = "UPDATE"
        to_copy["trans7"] = repo_root / "src" / "trans7_alphaqubit"
        to_copy["data"] = repo_root / "data" / "trans7_data"
        to_copy["arc_monitor"] = [
            repo_root / "arc_dashboard.py",
            repo_root / "arc_monitor.py",
            repo_root / "ARC_MONITOR.md"
        ]

    if not status["trans7"]:
        to_copy["trans7"] = repo_root / "src" / "trans7_alphaqubit"
    elif not status["whitelist"]:
        if status["missing_files"]:
            # not all files missing, copy just missing files
            to_copy["trans7_files"] = [
                repo_root / "src" / "trans7_alphaqubit" / f 
                for f in status["missing_files"]
            ]
        else:
            # whitelist failed but no file list returned - copy whole directory
            to_copy["trans7"] = repo_root / "src" / "trans7_alphaqubit"

    if not status["data"]:
        to_copy["data"] = repo_root / "data" / "trans7_data"
    
    if not status["arc_monitor"]:
        to_copy["arc_monitor"] = [
            repo_root / "arc_dashboard.py",
            repo_root / "arc_monitor.py",
            repo_root / "ARC_MONITOR.md"
        ]
    
    return to_copy


def _organize_arc_outputs(username: str, host: str, dry_run: bool = False):
    """
    REMOTE
    Uploads arc_organize.sh to ARC, runs it, then removes it.
    Moves logs -> ~/src/logs/ and checkpoints -> ~/src/checkpoints/.
    """
    print("="*60)
    print("ORGANIZING ARC LOGS AND CHECKPOINTS")
    print("="*60)

    script_local  = Path(__file__).parent / "arc_organize.sh"
    script_remote = "~/arc_organize.sh"
    ssh_target    = f"{username}@{host}"
    ssh_base      = ["ssh", "-4", "-o", "ControlPath=none", ssh_target]

    # Upload
    print("  Uploading arc_organize.sh ...")
    if not dry_run:
        subprocess.run(
            ["scp", "-4", "-o", "ControlPath=none",
             str(script_local), f"{ssh_target}:{script_remote}"],
            check=True
        )

    # Run
    dry_flag = "--dry-run" if dry_run else ""
    cmd = f"bash {script_remote} {dry_flag}".strip()
    print(f"  Running: {cmd}")
    if dry_run:
        print(f"  [dry-run] Would run: {cmd}")
    else:
        subprocess.run(ssh_base + [cmd], check=True)

    # Clean up
    print("  Removing arc_organize.sh from ARC ...")
    if not dry_run:
        subprocess.run(ssh_base + [f"rm -f {script_remote}"], check=True)


def copy_execution(manifest: dict, username: str, host: str, dry_run: bool = False):
    """
    LOCAL -> REMOTE
    After copy manifest determines what needs to be copied, and arc connect check 
    passes, use scp to copy the copy manifest to the ARC from LOCAL MACHINE
    """
    if not manifest:
        print("Nothing to copy.")
        return
    
    ssh_cmd = "ssh -4 -o ControlPath=none"

    # If update flag is set, wipe trans7 on ARC before transferring
    if "update" in manifest:
        print("="*60)
        print("UPDATE: wiping ~/src/trans7_alphaqubit on ARC")
        print("="*60)
        if not dry_run:
            subprocess.run(
                ["ssh", "-4", "-o", "ControlPath=none",
                 f"{username}@{host}",
                 "rm -rf ~/src/trans7_alphaqubit"],
                 check=True
            )
        else:
            print("  [dry-run] rm -rf ~/src/trans7_alphaqubit")

    dest_map = {
        "trans7":       f"{username}@{host}:~/src/trans7_alphaqubit",
        "trans7_files": f"{username}@{host}:~/src/trans7_alphaqubit",
        "data":         f"{username}@{host}:~/data/trans7_data/",
        "arc_monitor":  f"{username}@{host}:~/src/",
    }

    exclude = ["--exclude=__pycache__", "--exclude=*.pyc", "--exclude=.git"]

    for key, dest in dest_map.items():
        if key not in manifest:
            continue
        
        src = manifest[key]
        print(f"   rsync {key} -> {dest}")

        # arc_monitor is a list of files, others are single directory
        if isinstance(src, list):
            cmd = ["rsync", "-avz", "-e", ssh_cmd] + exclude + \
                  [str(p) for p in src] + [dest]
        else:
            cmd = ["rsync", "-avz", "-e", ssh_cmd] + exclude + \
                  [str(src) + "/",dest]
        
        if dry_run:
            print("  [dry-run]", " ".join(cmd))
        else:
            subprocess.run(cmd, check=True)




def main():
    """
    Fully setups environment on ARC necessary for running jobs.

    flags:
    --dry-run         : shows users what will happen on run, does not perform any action
    --arc-host        : by default is 'tinkercliffs2.arc.vt.edu
    --vtusername      : username of users to connect to ARC
    --update          : flags new update from Git, wipes trans7_alphaqubit and data from ARC and updates with new push
    """

    parser = argparse.ArgumentParser(
        description="Setup for running Trans7 on the ARC. " \
        "Launch with: python total_setup.py --vtusername USERNAME --host HOST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run",    action="store_true", help="Show what will happen without doing anything")
    parser.add_argument("--arc-host",   type=str, default="tinkercliffs2.arc.vt.edu", help="ARC login node hostname")
    parser.add_argument("--vtusername", type=str, required=True, help="Your VT username (e.g. cdw24)")
    parser.add_argument("--update",     action="store_true", help="Wipe trans7_alphaqubit on ARC and re-sync from local")

    args = parser.parse_args()

    trans7_status             = True
    trans7_whitelist_status   = True
    data_status               = True
    arc_monitor_status        = True
    update_status             = args.update

    arc_check = _arc_connect_check(args.vtusername, args.arc_host)
    
    if (arc_check == 0):
        print("Check credentials, quitting")
        quit()

    trans7_check = _trans7_check(args.vtusername, args.arc_host)

    missing_files = []
    if (trans7_check == 0):
        trans7_status = False
        trans7_whitelist_status = False
        print("Trans7 not present — skipping whitelist check.")
    else:
        trans7_whitelist_check = _trans7_whitelist(args.vtusername, args.arc_host)
        if (trans7_whitelist_check != 1):
            trans7_whitelist_status = False
            if isinstance(trans7_whitelist_check, list):
                missing_files = trans7_whitelist_check


    data_check = _data_check(args.vtusername, args.arc_host)

    if (data_check == 0):
        data_status = False
        print("Data status now False.")

    arc_monitor_check = _arc_monitor_check(args.vtusername, args.arc_host)

    if (arc_monitor_check == 0):
        arc_monitor_status = False
        print("Arc Monitor status now False.")
    
    print("="*60)
    print("Connections passed. Preparation check starting.")
    print("="*60)

    if (preparation() != 1):
        print("ERROR: Local preparation failed. Quitting.")
        quit()

    repo_root = _find_repo_root()
    _slurm_template_force(repo_root)
    
    status = {
        "trans7":         trans7_status,
        "whitelist":      trans7_whitelist_status,
        "missing_files":  missing_files,
        "data":           data_status,
        "arc_monitor":    arc_monitor_status,
        "update":         update_status
    }

    manifest = copy_manifest(status, repo_root)

    copy_execution(manifest, args.vtusername, args.arc_host, args.dry_run)

    _organize_arc_outputs(args.vtusername, args.arc_host, args.dry_run)

    print("="*60)
    print("Setup complete.")
    print("="*60)

if __name__ == "__main__":
    main()
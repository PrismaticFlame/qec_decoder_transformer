# Versions

## Local Machine

1. Move `arc_monitor.py` to ARC

```bash
scp arc_monitor.py ${USER}@tinkercliffs2.arc.vt.edu:~/src/
```

2. SSH into ARC

```bash
ssh ${USER}@tinkercliffs2.arc.vt.edu
```

3. Run `arc_monitor.py` with log directory pointer

```bash
/scratch/cdw24/conda_envs/alphaqubit_t7_env/bin/python ~/src/arc_monitor.py --job ${JOB_ID} --local --log-dir ~/src/trans7_alphaqubit/logs
```

## ARC Web Server

1. Move `arc_monitor.py` and `arc_dashboard.py` to ARC

```bash
scp arc_dashboard.py arc_monitor.py cdw24@tinkercliffs2.arc.vt.edu:~/src/
```

2. SSH into ARC and start server (supports multiple jobs)

```bash
ssh cdw24@tinkercliffs2.arc.vt.edu
python ~/src/arc_dashboard.py --jobs ${JOB_IDS}
```

3. Run tunnel in second terminal

```bash
ssh -L 5000:localhost:5000 -N ${USER}@tinkercliffs2.arc.vt.edu
```

4. Open browser

http://localhost:5000

## Server stays running after SSH disconnect version

1. Do first two commands with these changes:

```bash
ssh ${USER}@tinkercliffs2.arc.vt.edu
nohup python ~/src/arc_dashboard.py --jobs ${JOB_IDS} --log-dir ~/src/trans7_alphaqubit/logs > ~/src/arc_dashboard.log 2>&1 &
echo "Server PID: $!"
```

Can close that terminal entirely, server will remain running.

2. Can check in on the server with previous last command:

```bash
ssh -L 5000:localhost:5000 -N ${USER}@tinkercliffs2.arc.vt.edu
```

3. Open the browser page

http://localhost:5000

4. **To stop the server**

```bash
ssh ${USER}@tinkercliffs2.arc.vt.edu
pkill -f arc_dashboard.py
```


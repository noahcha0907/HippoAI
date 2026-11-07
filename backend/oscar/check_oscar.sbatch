#!/bin/bash
# Reports whether an Oscar GPU node can run the repair pipeline. Changes nothing.
# Submit from a login node:  sbatch check_oscar.sbatch
# Results go to:             oscar-check-<jobid>.out
#SBATCH -J hippo-check
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o oscar-check-%j.out

check_url() {
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$1")
  echo "$1 -> HTTP ${code} (000 means unreachable)"
}

echo "== Node =="
hostname
free -g | sed -n '2p'
nproc

echo; echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo; echo "== Outbound internet from this compute node =="
check_url https://huggingface.co
check_url https://api.moonshot.ai/v1/models
check_url https://pypi.org/simple/

echo; echo "== Sandbox options =="
echo "bwrap: $(command -v bwrap || echo 'not found')"
echo "apptainer: $(command -v apptainer || echo 'not found')"
echo "singularity: $(command -v singularity || echo 'not found')"
echo "max_user_namespaces: $(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo unknown)"
if unshare --user --map-root-user --net true 2>/dev/null; then
  echo "unprivileged user and network namespaces: WORK"
else
  echo "unprivileged user and network namespaces: BLOCKED"
fi
if command -v bwrap >/dev/null; then
  if bwrap --unshare-all --ro-bind /usr /usr --symlink usr/lib /lib --symlink usr/lib64 /lib64 \
       --symlink usr/bin /bin --proc /proc --dev /dev true 2>/dev/null; then
    echo "bwrap sandbox: WORKS"
  else
    echo "bwrap sandbox: FAILS"
  fi
fi

echo; echo "== Python =="
python3 --version
module avail python 2>&1 | tail -5

echo; echo "== Storage =="
checkquota 2>/dev/null || df -h "$HOME"
ls -d "$HOME/scratch" 2>/dev/null || echo "No ~/scratch"

echo; echo "== Partitions =="
sinfo -o '%P %a %l %G' 2>/dev/null | grep -i -E 'PARTITION|gpu'

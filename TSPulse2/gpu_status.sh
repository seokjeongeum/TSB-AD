#!/bin/bash

# A robust script to check nvidia-smi on all nodes where the user has running jobs.

echo "Searching for running jobs for user: $USER..."
echo

# 1. Get the unique list of nodes where you have running jobs.
NODES=$(squeue -u $USER -t RUNNING -h -o "%N" | sort -u)

# 2. Check if the NODES variable is empty (no running jobs found).
if [ -z "$NODES" ]; then
    echo "No running jobs found."
    exit 0
fi

# 3. Loop through each unique node and run nvidia-smi.
for NODE in $NODES
do
    echo "################################################################################"
    echo "## STATUS ON NODE: $NODE"
    echo "################################################################################"
    
    # Optional: List which of your jobs are on this specific node.
    echo "Your running jobs on this node:"
    squeue -u $USER -t RUNNING -w $NODE
    echo "---"

    # THE UPGRADES: Added flags to ssh for robustness and cleaner output.
    # -o ConnectTimeout=5 : Don't hang forever if a node is unresponsive.
    # -T : Disable tty allocation to prevent warnings.
    # -q : Suppress ssh banner messages.
    ssh -o ConnectTimeout=5 -T -q $NODE nvidia-smi
    echo
    echo
done

echo "Done."
#!/bin/bash

# --- Configuration ---
# The name for our tmux session
SESSION_NAME="sft_circuit_runners"

# An array holding the names of the scripts to run (change the folder of the parallel scripts as needed)
SCRIPTS=(
    "scripts/bash/parallel_sft_circuit/sft_circuit1.sh"
    "scripts/bash/parallel_sft_circuit/sft_circuit2.sh"
    "scripts/bash/parallel_sft_circuit/sft_circuit3.sh"
    "scripts/bash/parallel_sft_circuit/sft_circuit4.sh"
    "scripts/bash/parallel_sft_circuit/sft_circuit5.sh"
)
# The command to activate the virtual environment
ACTIVATE_CMD="source enveap/bin/activate"

# # Check if exactly 1 argument provided (language).
# if [ "$#" -ne 2 ]; then
#     echo "Error: Incorrect number of arguments."
#     echo "Usage: $0 <language (indo, eng, sunda)> <dataset_folder>"
#     exit 1 # Exit with a non-zero status to indicate an error
# fi

LANGUAGE="$1"
# Validate the language argument
if [[ "$LANGUAGE" != "indo" && "$LANGUAGE" != "eng" && "$LANGUAGE" != "sunda" ]]; then
    echo "Error: Invalid language specified. Use 'indo', 'eng', or 'sunda'."
    exit 1 # Exit with a non-zero status to indicate an error
fi

DATASET_FOLDER="$2"
# Validate the dataset folder argument
if [ -z "$DATASET_FOLDER" ]; then
    echo "Error: Dataset folder must be specified. Name a folder located in the hotel_dataset/{lang} directory."
    exit 1 # Exit with a non-zero status to indicate an error
fi

# If the first script contains 'sft_circuit.sh', we need to ensure it has the third argument TOPK_CIRCUIT
SFT_CIRCUIT_BOOL=false
if [[ "${SCRIPTS[0]}" == *"sft_circuit"* ]]; then
    TOPK_CIRCUIT="$3"
    if [ -z "$TOPK_CIRCUIT" ]; then
        echo "Error: TOPK_CIRCUIT must be specified for sft_circuit.sh."
        exit 1 # Exit with a non-zero status to indicate an error
    fi
    SFT_CIRCUIT_BOOL=true
fi

# --- Script Logic ---

# Check if the tmux session already exists. If it does, kill it.
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Found existing session '$SESSION_NAME'. Killing it now."
    tmux kill-session -t "$SESSION_NAME"
fi

echo "Creating new tmux session '$SESSION_NAME'..."

# Start a new detached tmux session. The first pane (index 0) is created automatically.
tmux new-session -d -s "$SESSION_NAME" -n "SFT Scripts"

tmux split-window -v -t "$SESSION_NAME:0.0"
tmux split-window -v -t "$SESSION_NAME:0.1"

# 2. Split the top row (pane 0) horizontally to create the second column.
tmux split-window -h -t "$SESSION_NAME:0.0"

# 3. Split the middle row (pane 1) horizontally to create the second column.
tmux split-window -h -t "$SESSION_NAME:0.1"

# Optional: You can resize the panes here if you want.
# For example, to make the right-most pane smaller:
# tmux resize-pane -R 20

# We don't need `select-layout` because we've built the layout manually.

# Loop through the panes and run the corresponding script in each one.
# The pane indices are now predictable: 0, 2, 1, 3, 4

for i in "${!SCRIPTS[@]}"; do
    PANE_INDEX=$i
    SCRIPT_NAME=${SCRIPTS[$i]}
    TARGET_PANE="$SESSION_NAME:0.$PANE_INDEX"
    
    # Construct the full command to be run in the pane (check for SFT_CIRCUIT_BOOL)
    echo $SFT_CIRCUIT_BOOL
    if [ "$SFT_CIRCUIT_BOOL" = true ]; then
        FULL_CMD="$ACTIVATE_CMD && bash ./${SCRIPT_NAME} ${LANGUAGE} ${DATASET_FOLDER} ${TOPK_CIRCUIT}"
    else
        FULL_CMD="$ACTIVATE_CMD && bash ./${SCRIPT_NAME} ${LANGUAGE} ${DATASET_FOLDER}"
    fi

    echo "Setting up pane $PANE_INDEX to run: ${SCRIPT_NAME}"
    
    # Send the commands to the pane.
    # The 'C-m' at the end simulates pressing the Enter key.
    tmux send-keys -t "$TARGET_PANE" "$FULL_CMD" C-m
done

# --- Attach to the Session ---
echo "Setup complete. Attaching to session '$SESSION_NAME'."
tmux attach-session -t "$SESSION_NAME"
#!/bin/bash

# Improved Two-Stage Experiment Script for Large Dataset
# Appropriate batch sizes for 40M row dataset

echo "============================================================"
echo "Improved Distributed Neural Network Experiment"
echo "============================================================"

# Create directories
mkdir -p results
mkdir -p logs
mkdir -p data

# Configuration
ACTIVATION="relu"
MAX_EPOCHS=500
PATIENCE=20
MIN_DELTA=0.0001
BASE_PROCESSES=4

# IMPROVED BATCH SIZES for 40M row dataset
# These are more appropriate for large-scale training
BATCH_SIZES=(1024 4096 16384 65536 262144) 

# Hidden sizes to test
HIDDEN_SIZES=(2)

# Process counts for scaling test
PROCESS_COUNTS=(2 4 6)

# Learning rates optimized for each activation
declare -A LEARNING_RATES
LEARNING_RATES["relu"]=0.001
LEARNING_RATES["sigmoid"]=0.01
LEARNING_RATES["tanh"]=0.005

# Parse command line arguments
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --preprocess          : Run data preprocessing first"
    echo "  --activation <act>    : Activation function (relu, sigmoid, tanh) [default: relu]"
    echo "  --base-processes <n>  : Processes for Stage 1 [default: 4]"
    echo "  --sample-size <n>     : Sample size for testing (-1 for full) [default: -1]"
    echo "  --help               : Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --preprocess --activation relu --sample-size 1000000"
    exit 1
}

PREPROCESS=false
SAMPLE_SIZE=-1

while [[ $# -gt 0 ]]; do
    case $1 in
        --preprocess)
            PREPROCESS=true
            shift
            ;;
        --activation)
            ACTIVATION="$2"
            shift 2
            ;;
        --base-processes)
            BASE_PROCESSES="$2"
            shift 2
            ;;
        --sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

LEARNING_RATE=${LEARNING_RATES[$ACTIVATION]}

# ============================================================
# STEP 0: Data Preprocessing (if requested)
# ============================================================
if [ "$PREPROCESS" = true ]; then
    echo "============================================================"
    echo "STEP 0: Data Preprocessing"
    echo "============================================================"
    
    # Check if raw data exists
    if [ ! -f "data/nytaxi2022.csv" ]; then
        echo "Error: data/nytaxi2022.csv not found!"
        echo "Please place the NYC taxi dataset in the data/ directory"
        exit 1
    fi
    
    echo "Running data preprocessing..."
    if [ $SAMPLE_SIZE -eq -1 ]; then
        python preprocess_data.py \
            --input data/nytaxi2022.csv \
            --output data/nytaxi2022_preprocessed.csv
    else
        python preprocess_data.py \
            --input data/nytaxi2022.csv \
            --output data/nytaxi2022_preprocessed.csv \
            --sample-size $SAMPLE_SIZE
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Preprocessing failed!"
        exit 1
    fi
    
    echo "Preprocessing complete!"
    echo ""
fi

# Check if preprocessed data exists
if [ ! -f "data/nytaxi2022_preprocessed.csv" ]; then
    echo "Error: Preprocessed data not found!"
    echo "Run with --preprocess flag to preprocess the data first"
    exit 1
fi

echo ""
echo "Experiment Configuration:"
echo "  Activation: $ACTIVATION"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max epochs: $MAX_EPOCHS (with early stopping)"
echo "  Patience: $PATIENCE epochs"
echo "  Min delta: $MIN_DELTA"
echo "  Batch sizes to test: ${BATCH_SIZES[@]}"
echo "  Hidden sizes to test: ${HIDDEN_SIZES[@]}"
echo ""

# Function to run experiment
run_experiment() {
    local activation=$1
    local batch_size=$2
    local hidden_size=$3
    local processes=$4
    local learning_rate=$5
    local stage=$6
    local exp_num=$7
    local total_exp=$8
    
    echo "[Stage $stage - Exp $exp_num/$total_exp] Running:"
    echo "  Activation: $activation"
    echo "  Batch size: $batch_size"
    echo "  Hidden size: $hidden_size"
    echo "  Processes: $processes"
    
    LOG_FILE="logs/stage${stage}_${activation}_h${hidden_size}_b${batch_size}_p${processes}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run improved neural network
    mpirun -np $processes python neural_network_mpi_improved.py \
        --data data/nytaxi2022_preprocessed.csv \
        --stats data/nytaxi2022_preprocessed_stats.json \
        --max-epochs $MAX_EPOCHS \
        --batch-size $batch_size \
        --activation $activation \
        --hidden-size $hidden_size \
        --learning-rate $learning_rate \
        --patience $PATIENCE \
        --min-delta $MIN_DELTA \
        2>&1 | tee "$LOG_FILE"
    
    echo "Completed Stage $stage - Exp $exp_num/$total_exp"
    echo ""
}

# ============================================================
# STAGE 1: Find optimal batch size and hidden size combination
# ============================================================
echo "============================================================"
echo "STAGE 1: Finding Optimal Batch Size and Hidden Size"
echo "============================================================"
echo "Testing batch sizes: ${BATCH_SIZES[@]}"
echo "Testing hidden sizes: ${HIDDEN_SIZES[@]}"
echo "Using $BASE_PROCESSES processes"
echo ""

experiment_count=0
total_stage1_experiments=$((${#BATCH_SIZES[@]} * ${#HIDDEN_SIZES[@]}))

best_rmse=999999
best_batch_size=0
best_hidden_size=0

echo "| Batch Size | Hidden Size | Train RMSE | Test RMSE | Test RMSE ($) | Epochs | Time (s) |"
echo "|------------|-------------|------------|-----------|---------------|--------|----------|"

for batch_size in "${BATCH_SIZES[@]}"; do
    for hidden_size in "${HIDDEN_SIZES[@]}"; do
        experiment_count=$((experiment_count + 1))
        
        # Run experiment
        run_experiment $ACTIVATION $batch_size $hidden_size $BASE_PROCESSES $LEARNING_RATE 1 $experiment_count $total_stage1_experiments
        
        # Find result file
        result_file=$(ls -t results/result_${ACTIVATION}_h${hidden_size}_b${batch_size}_p${BASE_PROCESSES}_*.json 2>/dev/null | head -1)
        
        if [ -f "$result_file" ]; then
            # Extract metrics using Python
            metrics=$(python3 -c "
import json
with open('$result_file', 'r') as f:
    data = json.load(f)
    print(f\"{data.get('train_rmse', 999):.6f}|{data.get('test_rmse', 999):.6f}|{data.get('test_rmse_denorm', 0):.2f}|{data.get('actual_epochs', 0)}|{data.get('training_time', 0):.2f}\")
" 2>/dev/null)
            
            if [ ! -z "$metrics" ]; then
                IFS='|' read -r train_rmse test_rmse test_rmse_denorm epochs time_val <<< "$metrics"
                printf "| %10d | %11d | %10s | %9s | $%12s | %6s | %8s |\n" \
                    $batch_size $hidden_size "$train_rmse" "$test_rmse" "$test_rmse_denorm" "$epochs" "$time_val"
                
                # Check if best
                is_better=$(awk -v t="$test_rmse" -v b="$best_rmse" 'BEGIN{print (t < b) ? 1 : 0}')
                if [ "$is_better" = "1" ]; then
                    best_rmse=$test_rmse
                    best_batch_size=$batch_size
                    best_hidden_size=$hidden_size
                fi
            fi
        fi
    done
done

echo ""
echo "★ Best configuration found:"
echo "  Batch size: $best_batch_size"
echo "  Hidden size: $best_hidden_size"
echo "  Test RMSE: $best_rmse"
echo ""

# ============================================================
# STAGE 2: Test parallel scaling with best configuration
# ============================================================
echo "============================================================"
echo "STAGE 2: Parallel Scaling Analysis"
echo "============================================================"
echo "Using best batch size: $best_batch_size"
echo "Using best hidden size: $best_hidden_size"
echo "Testing process counts: ${PROCESS_COUNTS[@]}"
echo ""

experiment_count=0
total_stage2_experiments=${#PROCESS_COUNTS[@]}

echo "| Processes | Training Time | Speedup | Efficiency | Test RMSE | Test RMSE ($) | Converged |"
echo "|-----------|---------------|---------|------------|-----------|---------------|-----------|"

baseline_time=0
for processes in "${PROCESS_COUNTS[@]}"; do
    experiment_count=$((experiment_count + 1))
    
    # Run experiment
    run_experiment $ACTIVATION $best_batch_size $best_hidden_size $processes $LEARNING_RATE 2 $experiment_count $total_stage2_experiments
    
    # Find result file
    result_file=$(ls -t results/result_${ACTIVATION}_h${best_hidden_size}_b${best_batch_size}_p${processes}_*.json 2>/dev/null | head -1)
    
    if [ -f "$result_file" ]; then
        # Extract metrics
        metrics=$(python3 -c "
import json
with open('$result_file', 'r') as f:
    data = json.load(f)
    print(f\"{data.get('training_time', 0):.2f}|{data.get('test_rmse', 999):.6f}|{data.get('test_rmse_denorm', 0):.2f}|{data.get('converged', False)}\")
" 2>/dev/null)
        
        if [ ! -z "$metrics" ]; then
            IFS='|' read -r training_time test_rmse test_rmse_denorm converged <<< "$metrics"
            
            # Set baseline
            if [ $processes -eq 1 ]; then
                baseline_time=$training_time
            fi
            
            # Calculate speedup and efficiency
            if (( $(echo "$baseline_time > 0" | bc -l) )); then
                speedup=$(echo "scale=2; $baseline_time / $training_time" | bc -l)
                efficiency=$(echo "scale=1; ($speedup / $processes) * 100" | bc -l)
            else
                speedup="N/A"
                efficiency="N/A"
            fi
            
            printf "| %9d | %13ss | %7s | %10s%% | %9s | $%12s | %9s |\n" \
                $processes "$training_time" "$speedup" "$efficiency" "$test_rmse" "$test_rmse_denorm" "$converged"
        fi
    fi
done

# ============================================================
# Final Summary Report
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE - FINAL SUMMARY"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Activation: $ACTIVATION"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max epochs: $MAX_EPOCHS (with early stopping)"
echo ""
echo "Key Findings:"
echo "  ✓ Optimal batch size: $best_batch_size"
echo "  ✓ Optimal hidden size: $best_hidden_size"
echo "  ✓ Best test RMSE: $best_rmse"
echo ""

# Generate report
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="results/improved_experiment_report_${ACTIVATION}_${TIMESTAMP}.txt"

{
    echo "Improved Neural Network Experiment Report"
    echo "Generated: $(date)"
    echo ""
    echo "Dataset: NYC Taxi 2022 (preprocessed)"
    echo "Sample size: $([ $SAMPLE_SIZE -eq -1 ] && echo "Full dataset" || echo "$SAMPLE_SIZE samples")"
    echo ""
    echo "Configuration:"
    echo "  Activation: $ACTIVATION"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Max epochs: $MAX_EPOCHS"
    echo "  Early stopping patience: $PATIENCE"
    echo "  Min delta: $MIN_DELTA"
    echo ""
    echo "Stage 1 - Hyperparameter Optimization:"
    echo "  Tested batch sizes: ${BATCH_SIZES[@]}"
    echo "  Tested hidden sizes: ${HIDDEN_SIZES[@]}"
    echo "  Best batch size: $best_batch_size"
    echo "  Best hidden size: $best_hidden_size"
    echo "  Best test RMSE: $best_rmse"
    echo ""
    echo "Stage 2 - Parallel Scaling:"
    echo "  Tested processes: ${PROCESS_COUNTS[@]}"
    echo "  Baseline time (1 process): ${baseline_time}s"
} > "$REPORT_FILE"

echo "Report saved to: $REPORT_FILE"
echo ""
echo "To visualize results, run:"
echo "  python analyze_improved_results.py"
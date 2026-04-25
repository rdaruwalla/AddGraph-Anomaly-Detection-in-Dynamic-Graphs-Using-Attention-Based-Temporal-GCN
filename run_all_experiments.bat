@echo off
setlocal

echo ========================================
echo Running AddGraph experiment batch
echo ========================================

if not exist ".\capstone_addgraph\results\setting_a" mkdir ".\capstone_addgraph\results\setting_a"
if not exist ".\capstone_addgraph\results\setting_b" mkdir ".\capstone_addgraph\results\setting_b"
if not exist ".\capstone_addgraph\results\validation_4file" mkdir ".\capstone_addgraph\results\validation_4file"

echo.
echo ========================================
echo SETTING A - seed 42 - addgraph
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name addgraph --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 2 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_a\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING A - seed 42 - temporal_no_attention
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name temporal_no_attention --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 2 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_a\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING A - seed 42 - static_gcn
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name static_gcn --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_a\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING A - seed 43 - addgraph
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name addgraph --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 43 --hidden_dim 32 --gcn_layers 2 --window_size 2 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_a\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING A - seed 43 - temporal_no_attention
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name temporal_no_attention --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 43 --hidden_dim 32 --gcn_layers 2 --window_size 2 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_a\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING A - seed 43 - static_gcn
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name static_gcn --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 43 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_a\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING B - seed 42 - addgraph
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name addgraph --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_b\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING B - seed 42 - temporal_no_attention
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name temporal_no_attention --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_b\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING B - seed 42 - static_gcn
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name static_gcn --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_b\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING B - seed 43 - addgraph
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name addgraph --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 43 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_b\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING B - seed 43 - temporal_no_attention
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name temporal_no_attention --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 43 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_b\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SETTING B - seed 43 - static_gcn
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name static_gcn --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 43 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\setting_b\
if errorlevel 1 goto :error

echo.
echo ========================================
echo VALIDATION 4FILE - seed 42 - addgraph
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name addgraph --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 2 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv,Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\validation_4file\
if errorlevel 1 goto :error

echo.
echo ========================================
echo VALIDATION 4FILE - seed 42 - temporal_no_attention
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name temporal_no_attention --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 2 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv,Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\validation_4file\
if errorlevel 1 goto :error

echo.
echo ========================================
echo VALIDATION 4FILE - seed 42 - static_gcn
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name static_gcn --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv,Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\validation_4file\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SELECTIVE SAMPLING ON - seed 42 - temporal_no_attention
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name temporal_no_attention --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 2 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --output_dir .\capstone_addgraph\results\selective_sampling_on\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SELECTIVE SAMPLING OFF - seed 42 - temporal_no_attention
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name temporal_no_attention --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 2 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\selective_sampling_off\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SELECTIVE SAMPLING ON - seed 42 - static_gcn
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name static_gcn --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --output_dir .\capstone_addgraph\results\selective_sampling_on\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SELECTIVE SAMPLING OFF - seed 42 - static_gcn
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name static_gcn --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 1 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\selective_sampling_off\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SELECTIVE SAMPLING ON - seed 42 - addgraph
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name addgraph --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 2 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --output_dir .\capstone_addgraph\results\selective_sampling_on\
if errorlevel 1 goto :error

echo.
echo ========================================
echo SELECTIVE SAMPLING OFF - seed 42 - addgraph
echo ========================================
python capstone_addgraph/experiments/run_experiment.py --data_dir CICIDS2017 --model_name addgraph --bucket 5min --epochs 3 --train_ratio 0.7 --device cuda --seed 42 --hidden_dim 32 --gcn_layers 2 --window_size 2 --max_rows_per_file 300000 --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv" --disable_training_pair_filter --output_dir .\capstone_addgraph\results\selective_off_4file\
if errorlevel 1 goto :error

echo.
echo ========================================
echo ALL RUNS FINISHED SUCCESSFULLY
echo ========================================
pause
exit /b 0

:error
echo.
echo ========================================
echo A command failed. Stopping batch file.
echo ========================================
pause
exit /b 1
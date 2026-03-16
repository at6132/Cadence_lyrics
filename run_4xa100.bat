@echo off
cd /d "%~dp0"
set LYRIC_DEVICE=4xa100
accelerate launch --config_file configs/accelerate_4xa100.yaml finetune.py
pause

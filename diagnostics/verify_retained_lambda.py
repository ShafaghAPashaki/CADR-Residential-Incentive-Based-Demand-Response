"""Offline integrity check for frozen original λc, no calibration or training."""
from pathlib import Path
import json
import sys
import yaml

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from utils.final_run_protocol import enforce_lambda_calibration, config_sha256  # noqa: E402

for seed in (0,1,2):
    cfg=yaml.safe_load((root/f"config_seed_{seed}.yaml").read_text(encoding="utf-8"))
    audit=enforce_lambda_calibration(cfg)
    if int(cfg['DDQN']['episodes']) != 2500:
        raise RuntimeError(f"Seed {seed}: final episode count changed")
    print(json.dumps({
        'seed':seed,
        'lambda_c':cfg['environment']['symmetric_tracking']['lambda_c'],
        'lambda_source':audit['lambda_source'],
        'config_sha256':config_sha256(cfg),
        'forecast_sha256':audit['training_forecast_file_sha256'],
        'retention_status':audit['status'],
        'episodes':cfg['DDQN']['episodes'],
    },sort_keys=True))
print('PASS: three seed configs preserve original lambda_c and frozen forecast')

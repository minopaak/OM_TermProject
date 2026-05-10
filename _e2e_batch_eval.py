"""E2E 테스트: 1 SKU 배치 + 평가. 보고서의 검토 권고를 자동 적용."""
from __future__ import annotations

import sys
import time
from datetime import date

from src.agents.batch import BatchCase, run_batch
from src.evaluation.metrics import evaluate_batch_run

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_INPUT_END = date(2016, 1, 28)
cases = [
    # POC: single SKU manually verified.
    # FOODS_3_295_CA_1 — full 5-year history, daily avg 14.0, 2016-01 input
    # window stable (0 zeros). PresidentsDay history shows median ratio 1.56
    # (n=5). Most importantly, actual 2016 forecast period shows clear
    # PresidentsDay spike (49 on 2/15 vs typical Mon ~13-17). Clean POC case.
    BatchCase(sku_id="FOODS_3_295_CA_1", input_end_date=_INPUT_END),
]

print(f"=== Batch 시작 ({len(cases)} case) ===")
t0 = time.time()
result = run_batch(cases, run_id="poc_FOODS_3_295_CA_1_v54_lstm_dow")
print(f"\n총 {time.time()-t0:.1f}s")

print(f"\n=== 평가 ===")
eval_df = evaluate_batch_run(result.out_dir)

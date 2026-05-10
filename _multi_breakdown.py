"""Per-SKU breakdown of multi-SKU batch."""
import pandas as pd

run_dir = "data/runs/poc_FOODS_3_295_CA_1_v52_no_scenario"
df = pd.read_parquet(f"{run_dir}/evaluation.parquet")

print(f"{'sku_id':<22} | {'actual':>6} | {'base':>6} | {'final':>6} | "
      f"{'tot_err_b':>9} | {'tot_err_f':>9} | {'tot_FVA':>8} | "
      f"{'MAE_FVA':>8} | {'sMAPE_FVA':>10}")
print("-" * 130)
for _, r in df.iterrows():
    actual = r["actual_total"]
    base = r["baseline_total"]
    final = r["final_total"]
    base_te = abs(base - actual)
    final_te = abs(final - actual)
    tot_fva = base_te - final_te  # positive = 개선
    mae_fva = r["fva_mae"]
    smape_fva = r.get("fva_smape", float("nan"))
    print(
        f"{r['sku_id']:<22} | {actual:>6.0f} | {base:>6.0f} | {final:>6.0f} | "
        f"{base_te:>11.0f} | {final_te:>11.0f} | {tot_fva:>+8.0f} | "
        f"{mae_fva:>+8.2f} | {smape_fva:>+9.2f}%p"
    )

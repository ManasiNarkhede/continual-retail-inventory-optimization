import json

with open(".sisyphus/evidence/t0b_coverage_diagnosis_raw.json") as f:
    data = json.load(f)

print("Per-quantile coverage:")
for q, cov in data["coverage"]["per_quantile"].items():
    print(f"  q={q}: {cov:.4%}")

print(
    f'\nMonotonicity violations: {data["monotonicity"]["violations"]}/'
    f'{data["monotonicity"]["total_rows"]} ({data["monotonicity"]["violation_rate"]:.2%})'
)
print(f'Median bias: {data["median_bias"]:.2f}')

print("\nQuantile spread means:")
for q, stats in data["spreads"].items():
    print(
        f'  q={q}: mean={stats["mean"]:.2f}, std={stats["std"]:.2f}, '
        f'min={stats["min"]:.2f}, max={stats["max"]:.2f}'
    )

print("\nTarget stats:")
for k, v in data["data"]["target_stats"].items():
    print(f"  {k}={v:.2f}")

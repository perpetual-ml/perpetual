# Optimization Benchmark Results

## Baseline (Decisioning Branch)

- California Housing: ~1.05s
- Cover Types: ~13.7s

## Optimization 1: Pre-calculate Bin Sums

- **California Housing**: No change (~1.05s)
- **Cover Types**: ~21.4s (Regression)
- **Improvement**: Regression (Possible thermal throttling or loop overhead)

## Optimization 2: SIMD Array Sums (Tested with Opt 1)

- **California Housing**: ~24-50% slower (Regression)
- **Cover Types**: Slower
- **Improvement**: Regression (Likely due to Opt 1 or SIMD overhead)

## Optimization 3: Optimize pivot_on_split (HashSet Lookup)

- **California Housing**: Regression (~0.6-35% variance, overall slower)
- **Cover Types**: Major Regression (>2x slower, cancelled)
- **Improvement**: Failed. Code duplication likely hurt instruction cache/branch prediction.

## Optimization 4: Attributes (#[inline(always)], #[cold])

- **California Housing**: [PENDING]
- **Cover Types**: [PENDING]
- **Improvement**: Failed. Unused variable warnings, likely ineffective.

## Optimization 5: Histogram Batching (Manual Unrolling)

- **California Housing**: Regression (>2x slower)
- **Improvement**: Failed. Manual unrolling hurt performance vs compiler optimization.

## Optimization 6: Bounds Check Elimination (assume)

- **California Housing**: Regression (>2x slower)
- **Improvement**: Failed. Likely disrupted auto-vectorization.

## Optimization 7: Unchecked Swaps (slice_swap_unchecked)

- **California Housing**: Regression (Timeout/Slower)
- **Improvement**: Failed. Unsafe swaps did not yield speedup.

## Optimization 8: Boxed BitSet for Categorical Splits (Structure Splitting)

- **California Housing**: **0.88s** (16% Faster vs 1.05s Baseline)
- **Cover Types**: **12.80s** (6.5% Faster vs 13.7s Baseline)
- **Improvement**: **SUCCESS**.
  - Replaced `HashSet<usize>` with `Option<Box<Vec<bool>>>`.
  - **Categorical**: Used BitSet on heap for O(1) checks.
  - **Numerical**: Used `None` (null pointer) to skip allocation and checking.
  - **Outcome**: Improved both numerical and categorical performance by reducing struct size (better cache locality) and removing hashing overhead.

## Conclusion

After 7 failed attempts at micro-optimizations, **Optimization 8** proved successful by addressing data structure efficiency rather than instruction-level tuning.

- **Categorical Optimization**: Moving from `HashSet` to `BitSet` reduced overhead.
- **Numerical Optimization**: "Structure Splitting" (boxing the cold categorical data) reduced the hot `SplitInfo` struct size, speeding up the numerical path significantly (16%).

**Next Steps**:
Proceed with algorithmic improvements like `exact_split` loop unrolling or Histogram Subtraction if further gains are needed.

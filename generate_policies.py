#!/usr/bin/env python3
"""
generate_policies.py

Standalone CLI script to generate synthetic policy data for the ULP model.

Usage:
    python generate_policies.py [--num-policies N] [--seed S] [--output FILE]

Examples:
    python generate_policies.py                           # Default: 1M policies
    python generate_policies.py --num-policies 10000000   # 10M policies
    python generate_policies.py --seed 123 --num-policies 5000000

The script will create policy_data/ directory and save the parquet file there.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def generate_policy_data(n_policies: int, seed: int = None) -> pd.DataFrame:
    """
    Generate synthetic policy data using fully vectorized NumPy operations.
    
    Parameters
    ----------
    n_policies : int
        Number of policies to generate
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with policy data
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"Generating {n_policies:,} policies...")
    start_time = time.time()
    
    data = {}
    
    # 1. policy_id: Sequential from 1 to n_policies
    data['policy_id'] = np.arange(1, n_policies + 1, dtype=np.int32)
    
    # 2. age_at_entry: Random integer between 0 and 65
    data['age_at_entry'] = np.random.randint(0, 66, size=n_policies, dtype=np.int32)
    
    # 3. sex: Random 0 (male) or 1 (female)
    data['sex'] = np.random.randint(0, 2, size=n_policies, dtype=np.int32)
    
    # 4. pol_term: 90 - age_at_entry
    data['pol_term'] = (90 - data['age_at_entry']).astype(np.int32)
    
    # 5. prem_term: Random integer from 20 to pol_term
    prem_term = np.zeros(n_policies, dtype=np.int32)
    for pt in range(20, 91):
        mask = data['pol_term'] == pt
        count = mask.sum()
        if count > 0:
            prem_term[mask] = np.random.randint(20, pt + 1, size=count)
    data['prem_term'] = prem_term
    
    # 6. prem_freq: Random choice from [12, 6, 3, 1]
    prem_freq_choices = np.array([12, 6, 3, 1], dtype=np.int32)
    data['prem_freq'] = np.random.choice(prem_freq_choices, size=n_policies).astype(np.int32)
    
    # 7 & 8. acp and sum_assd
    acp_multiplier = np.random.randint(6, 101, size=n_policies, dtype=np.int32)
    data['acp'] = (acp_multiplier * 1_000_000).astype(np.float64)
    
    sum_assd_multiplier = np.random.randint(5, 21, size=n_policies, dtype=np.int32)
    data['sum_assd'] = (sum_assd_multiplier * data['acp']).astype(np.float64)
    
    # 9. db_opt: Death benefit option (1=basic, 2=escalating)
    data['db_opt'] = np.random.randint(1, 3, size=n_policies, dtype=np.int32)
    
    # 10. atp: Random int [0, 20] * 1_000_000
    atp_multiplier = np.random.randint(0, 21, size=n_policies, dtype=np.int32)
    data['atp'] = (atp_multiplier * 1_000_000).astype(np.float64)
    
    # 11. topup_term: Random int [0, pol_term]
    topup_term = np.zeros(n_policies, dtype=np.int32)
    for pt in range(0, 91):
        mask = data['pol_term'] == pt
        count = mask.sum()
        if count > 0:
            topup_term[mask] = np.random.randint(0, pt + 1, size=count)
    data['topup_term'] = topup_term
    
    # 12. topup_freq: Random choice from [12, 6, 3, 1]
    data['topup_freq'] = np.random.choice(prem_freq_choices, size=n_policies).astype(np.int32)
    
    # 13. mort_loading: 95% zero, 5% random int [0, 100]
    mort_loading = np.zeros(n_policies, dtype=np.float64)
    mask = np.random.random(n_policies) < 0.05
    mort_loading[mask] = np.random.randint(0, 101, size=mask.sum()).astype(np.float64)
    data['mort_loading'] = mort_loading
    
    # 14. init_pols_if: Always 1
    data['init_pols_if'] = np.ones(n_policies, dtype=np.float64)
    
    df = pd.DataFrame(data)
    
    elapsed = time.time() - start_time
    print(f"✓ Generated {n_policies:,} policies in {elapsed:.2f} seconds")
    print(f"  ({n_policies / elapsed:,.0f} policies/sec)")
    
    return df


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate generated policy data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Generated policy data
        
    Returns
    -------
    bool
        True if all validation checks pass
    """
    print("\nValidating policy data...")
    
    checks_passed = 0
    checks_total = 0
    
    def check(condition, message):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            print(f"  ✓ {message}")
            checks_passed += 1
            return True
        else:
            print(f"  ✗ FAILED: {message}")
            return False
    
    # Run all checks
    check(df['age_at_entry'].min() >= 0 and df['age_at_entry'].max() <= 65,
          f"age_at_entry: range [{df['age_at_entry'].min()}, {df['age_at_entry'].max()}]")
    
    check((df['pol_term'] == 90 - df['age_at_entry']).all(),
          f"pol_term = 90 - age_at_entry")
    
    check((df['prem_term'] >= 20).all() and (df['prem_term'] <= df['pol_term']).all(),
          f"prem_term: all in [20, pol_term]")
    
    check(df['prem_freq'].isin([1, 3, 6, 12]).all(),
          f"prem_freq: all in {{1, 3, 6, 12}}")
    
    check((df['sum_assd'] > 0).all(),
          f"sum_assd: all positive")
    
    check(df['db_opt'].isin([1, 2]).all(),
          f"db_opt: all in {{1, 2}}")
    
    check((df['acp'] >= 6e6).all() and (df['acp'] <= 100e6).all(),
          f"acp: all in [6M, 100M]")
    
    check((df['atp'] >= 0).all() and (df['atp'] <= 20e6).all(),
          f"atp: all in [0, 20M]")
    
    check((df['topup_term'] >= 0).all() and (df['topup_term'] <= df['pol_term']).all(),
          f"topup_term: all in [0, pol_term]")
    
    check(df['topup_freq'].isin([1, 3, 6, 12]).all(),
          f"topup_freq: all in {{1, 3, 6, 12}}")
    
    pct_nonzero = (df['mort_loading'] > 0).sum() / len(df) * 100
    check(4.5 <= pct_nonzero <= 5.5,
          f"mort_loading: {pct_nonzero:.2f}% non-zero (target ~5%)")
    
    check((df['init_pols_if'] == 1.0).all(),
          f"init_pols_if: all ones")
    
    check(df.isnull().sum().sum() == 0,
          f"No null values")
    
    check(df['policy_id'].nunique() == len(df),
          f"All policy_ids unique")
    
    all_passed = checks_passed == checks_total
    print(f"\n✓ {checks_passed}/{checks_total} validation checks passed")
    
    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic policy data for the ULP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_policies.py                          # 1M policies (default)
  python generate_policies.py --num-policies 10000000  # 10M policies
  python generate_policies.py --seed 123               # With custom seed
  python generate_policies.py --num-policies 3000000 --seed 456 --output custom.parquet
        """
    )
    
    parser.add_argument(
        "--num-policies", "-n",
        type=int,
        default=1_000_000,
        help="Number of policies to generate (default: 1,000,000)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="policies.parquet",
        help="Output filename (default: policies.parquet)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_policies <= 0:
        print("ERROR: --num-policies must be positive")
        sys.exit(1)
    
    # Create policy_data directory if it doesn't exist
    policy_dir = Path("policy_data")
    policy_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = policy_dir / args.output
    
    try:
        print("="*80)
        print("POLICY DATA GENERATOR FOR ULP MODEL")
        print("="*80)
        print(f"Parameters:")
        print(f"  Number of policies: {args.num_policies:,}")
        print(f"  Random seed: {args.seed}")
        print(f"  Output path: {output_path}")
        print()
        
        # Generate data
        df = generate_policy_data(args.num_policies, seed=args.seed)
        
        # Validate
        if not validate_dataframe(df):
            print("\nERROR: Validation failed!")
            sys.exit(1)
        
        # Show statistics
        print("\n" + "="*80)
        print("DATA SUMMARY")
        print("="*80)
        print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print()
        
        # Save to parquet
        print("Saving to parquet...")
        save_start = time.time()
        df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        save_elapsed = time.time() - save_start
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        
        print("="*80)
        print("SUCCESS")
        print("="*80)
        print(f"✓ Saved to: {output_path}")
        print(f"✓ File size: {file_size_mb:.2f} MB")
        print(f"✓ Save time: {save_elapsed:.2f} seconds")
        print(f"✓ Rows: {len(df):,}")
        print(f"✓ Columns: {len(df.columns)}")
        print()
        print("Ready to use with the ULP model!")
        print("Update your config.yaml:")
        print(f"  policy_inputs_file: policy_data/{args.output}")
        
    except KeyboardInterrupt:
        print("\n\nGeneration cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

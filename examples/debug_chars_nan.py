# Debug script for chars NaN issue
# Run this BEFORE calling sf.fit() and AFTER creating sf

import numpy as np
import pandas as pd
import PyBondLab as pbl

# Assuming you have:
# - data: your DataFrame
# - sf: your StrategyFormation object (before calling fit())

def debug_chars_issue(data, sf):
    """Debug why characteristics might be returning all NaN."""

    print("=" * 60)
    print("DEBUG: Characteristics NaN Issue")
    print("=" * 60)

    # 1. Check column names
    print("\n1. COLUMN NAMES CHECK")
    print("-" * 40)
    char_col = sf.chars[0] if sf.chars else None
    if char_col:
        print(f"Looking for char column: '{char_col}'")
        print(f"Column in data.columns: {char_col in data.columns}")

        # Check for similar column names (case/whitespace issues)
        similar = [c for c in data.columns if char_col.lower().strip() in c.lower()]
        print(f"Similar columns: {similar}")

        # Check column name bytes
        print(f"Column name bytes: {[ord(c) for c in char_col]}")
        for col in data.columns:
            if char_col.lower() in col.lower():
                print(f"  Found '{col}': bytes = {[ord(c) for c in col]}")

    # 2. Check data types and values
    print("\n2. DATA TYPE AND VALUES CHECK")
    print("-" * 40)
    if char_col and char_col in data.columns:
        col_data = data[char_col]
        print(f"dtype: {col_data.dtype}")
        print(f"shape: {col_data.shape}")
        print(f"null count: {col_data.isnull().sum()} ({col_data.isnull().mean()*100:.1f}%)")
        print(f"non-null count: {col_data.notna().sum()}")
        print(f"unique values (non-null): {col_data.dropna().nunique()}")

        # Check for inf values
        if np.issubdtype(col_data.dtype, np.floating):
            inf_count = np.isinf(col_data).sum()
            print(f"inf count: {inf_count}")

        # Value range
        print(f"min: {col_data.min()}, max: {col_data.max()}")
        print(f"sample values:\n{col_data.dropna().head(10).tolist()}")

    # 3. Check required_cols
    print("\n3. REQUIRED COLUMNS CHECK")
    print("-" * 40)
    if hasattr(sf, 'required_cols'):
        print(f"required_cols: {sf.required_cols}")
        print(f"char_col in required_cols: {char_col in sf.required_cols}")
    else:
        print("required_cols not yet built (call fit() first)")

    # 4. Run fit and capture intermediate data
    print("\n4. RUNNING FIT WITH DEBUG...")
    print("-" * 40)

    # Monkey-patch to capture intermediate data
    original_form_single = sf._form_single_period

    debug_data = {
        'It1m_cols': [],
        'It1m_aug_cols': [],
        'It1m_aug_shape': [],
        'char_values_sample': [],
        'char_nan_count': [],
        'result_chars': []
    }

    call_count = [0]

    def debug_form_single(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 3:  # Only debug first 3 calls
            It0, It1, It1m = args[0], args[1], args[2]

            debug_data['It1m_cols'].append(list(It1m.columns) if not It1m.empty else [])

            # Check if char_col is in It1m
            if char_col and char_col in It1m.columns:
                char_vals = It1m[char_col]
                debug_data['char_nan_count'].append(char_vals.isnull().sum())
                debug_data['char_values_sample'].append(
                    char_vals.dropna().head(5).tolist() if char_vals.notna().any() else []
                )
            else:
                debug_data['char_nan_count'].append('COLUMN NOT FOUND')
                debug_data['char_values_sample'].append([])

        result = original_form_single(*args, **kwargs)

        if call_count[0] <= 3:
            if result.get('chars_ew') is not None:
                chars_df = result['chars_ew']
                debug_data['result_chars'].append({
                    'shape': chars_df.shape,
                    'columns': list(chars_df.columns),
                    'null_count': chars_df.isnull().sum().to_dict(),
                    'values': chars_df.to_dict() if chars_df.size < 50 else 'too large'
                })
            else:
                debug_data['result_chars'].append(None)

        return result

    sf._form_single_period = debug_form_single

    try:
        # Run fit
        results = sf.fit(IDvar='cusip', RETvar='ret_vw_bgn', VWvar='mcap_e',
                        RATINGvar='spc_rat')

        print("\n5. DEBUG DATA FROM FIRST 3 PERIODS")
        print("-" * 40)
        print(f"It1m columns (first period): {debug_data['It1m_cols'][0] if debug_data['It1m_cols'] else 'N/A'}")
        print(f"Char '{char_col}' NaN counts: {debug_data['char_nan_count']}")
        print(f"Char values samples: {debug_data['char_values_sample']}")
        print(f"Result chars info: {debug_data['result_chars']}")

        # Check final results
        print("\n6. FINAL RESULTS CHECK")
        print("-" * 40)
        chars_ew, chars_vw = results.get_characteristics()
        if chars_ew:
            for c, df in chars_ew.items():
                print(f"Char '{c}':")
                print(f"  Shape: {df.shape}")
                print(f"  All NaN: {df.isnull().all().all()}")
                print(f"  NaN count per column: {df.isnull().sum().to_dict()}")
                print(f"  First non-NaN values:\n{df.dropna(how='all').head(3)}")

        return results

    finally:
        sf._form_single_period = original_form_single

# Example usage:
# results = debug_chars_issue(data, sf)

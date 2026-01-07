#!/usr/bin/env python3
"""
================================================================================
ACCURACY COMPARISON SCRIPT
================================================================================

Compares predictions from all data sources against Meteocontrol actual 
generation data. Provides terminal, graphical, and text report outputs.

INPUT FILES:
------------
  data/meteocontrol_actual.csv          - Actual generation (semicolon-delimited)
  output/predictions_nasa_power.csv     - NASA POWER predictions
  output/predictions_solcast.csv        - Solcast predictions
  output/predictions_openmeteo.csv      - Open-Meteo predictions
  output/predictions_era5.csv           - ERA5 predictions

OUTPUT FILES:
-------------
  output/accuracy_comparison.png        - Bar chart comparing all sources
  output/daily_comparison.png           - Line chart: actual vs predicted
  output/accuracy_report.txt            - Detailed text report
  output/accuracy_report.json           - JSON format (with --json flag)

USAGE:
------
  python compare_accuracy.py                           # Default comparison
  python compare_accuracy.py --actual custom.csv       # Custom Meteocontrol file
  python compare_accuracy.py --days 1-15               # Filter by day range
  python compare_accuracy.py --no-plot                 # Skip graphical output
  python compare_accuracy.py --json                    # Output as JSON

================================================================================
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# =====================================================================
# CONFIGURATION
# =====================================================================

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"

# Granularity directories
HOURLY_DIR = OUTPUT_DIR / "hourly"
DAILY_DIR = OUTPUT_DIR / "daily"
MONTHLY_DIR = OUTPUT_DIR / "monthly"

# Prediction file names
PREDICTION_FILENAMES = {
    "NASA POWER": "predictions_nasa_power.csv",
    "Solcast": "predictions_solcast.csv",
    "Open-Meteo": "predictions_openmeteo.csv",
    "ERA5": "predictions_era5.csv",
}

def get_prediction_files(granularity="hourly"):
    """Get prediction file paths for the specified granularity."""
    if granularity == "hourly":
        folder = HOURLY_DIR
    elif granularity == "daily":
        folder = DAILY_DIR
    elif granularity == "monthly":
        folder = MONTHLY_DIR
    else:
        raise ValueError(f"Unknown granularity: {granularity}")
    
    return {name: folder / filename for name, filename in PREDICTION_FILENAMES.items()}

# Default Meteocontrol file
DEFAULT_METEOCONTROL = DATA_DIR / "meteocontrol_actual.csv"


# =====================================================================
# DATA LOADING
# =====================================================================

def load_meteocontrol(path):
    """
    Load Meteocontrol CSV with semicolon delimiter.
    
    Expected columns: Category, Power, Target range (low), Target range (high), 
                      Losses, POA irradiation (satellite)
    
    Returns:
        DataFrame with 'day' as index and cleaned columns
    """
    print(f"Loading Meteocontrol data: {path}")
    
    df = pd.read_csv(path, sep=";", quotechar='"')
    
    # Parse Category as day number (handle both quoted string and numeric)
    if df["Category"].dtype == object:
        df["day"] = pd.to_numeric(df["Category"].str.strip('"'), errors="coerce").astype(int)
    else:
        df["day"] = df["Category"].astype(int)
    
    # Rename columns for easier access
    df = df.rename(columns={
        "Power": "actual_kwh",
        "Target range (low)": "target_low",
        "Target range (high)": "target_high",
        "Losses": "losses",
        "POA irradiation (satellite)": "poa_irradiation"
    })
    
    # Set day as index
    df = df.set_index("day")
    
    # Drop rows with missing actual values
    df = df.dropna(subset=["actual_kwh"])
    
    # Convert numeric columns
    for col in ["actual_kwh", "target_low", "target_high", "losses", "poa_irradiation"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    print(f"  Loaded {len(df)} days of actual data")
    print(f"  Day range: {df.index.min()} to {df.index.max()}")
    
    return df


def load_predictions(path, source_name, granularity="hourly"):
    """
    Load prediction CSV for a given granularity.
    
    Args:
        path: Path to CSV file
        source_name: Name for logging
        granularity: One of "hourly", "daily", "monthly"
    
    Returns:
        DataFrame with predictions indexed by day of month (or month number)
        None if file doesn't exist or is empty
    """
    if not path.exists():
        print(f"  ⚠️ {source_name}: File not found - {path}")
        return None
    
    try:
        if granularity == "hourly":
            # Hourly files have 'timestamp' and 'predicted_power_kw'
            df = pd.read_csv(path, parse_dates=["timestamp"])
            
            if df.empty:
                print(f"  ⚠️ {source_name}: File is empty")
                return None
            
            # Extract day from timestamp
            df["day"] = df["timestamp"].dt.day
            
            # Aggregate to daily totals (sum of hourly kW = kWh for 1-hour intervals)
            daily = df.groupby("day").agg({
                "predicted_power_kw": "sum",
                "timestamp": "count"  # Count hours to detect incomplete days
            }).rename(columns={
                "predicted_power_kw": "predicted_kwh",
                "timestamp": "hours_count"
            })
            
            # Flag incomplete days (less than 20 hours of data)
            daily["complete"] = daily["hours_count"] >= 20
            
            print(f"  ✓ {source_name}: {len(daily)} days loaded, {daily['complete'].sum()} complete")
            return daily
            
        elif granularity == "daily":
            # Daily files have 'date' and 'predicted_energy_kwh'
            df = pd.read_csv(path, parse_dates=["date"])
            
            if df.empty:
                print(f"  ⚠️ {source_name}: File is empty")
                return None
            
            # Extract day from date
            df["day"] = df["date"].dt.day
            
            # Rename to standard column
            daily = df.set_index("day")[["predicted_energy_kwh", "hours_count"]].copy()
            daily = daily.rename(columns={"predicted_energy_kwh": "predicted_kwh"})
            daily["complete"] = daily["hours_count"] >= 20
            
            print(f"  ✓ {source_name}: {len(daily)} days loaded, {daily['complete'].sum()} complete")
            return daily
            
        elif granularity == "monthly":
            # Monthly files have 'month' and 'predicted_energy_kwh'
            df = pd.read_csv(path, parse_dates=["month"])
            
            if df.empty:
                print(f"  ⚠️ {source_name}: File is empty")
                return None
            
            # Use month as index
            df["month_num"] = df["month"].dt.month
            
            # Rename to standard column
            monthly = df.set_index("month_num")[["predicted_energy_kwh", "days_count"]].copy()
            monthly = monthly.rename(columns={
                "predicted_energy_kwh": "predicted_kwh",
                "days_count": "hours_count"  # Reuse for count display
            })
            monthly["complete"] = True  # Monthly always considered complete
            
            print(f"  ✓ {source_name}: {len(monthly)} months loaded")
            return monthly
        
    except Exception as e:
        print(f"  ⚠️ {source_name}: Error loading - {e}")
        return None


# =====================================================================
# METRICS CALCULATION
# =====================================================================

def calculate_metrics(actual, predicted):
    """
    Calculate accuracy metrics for aligned actual vs predicted data.
    
    Args:
        actual: Series of actual values
        predicted: Series of predicted values (aligned with actual)
    
    Returns:
        dict with MAE, RMSE, MAPE, Bias, R², count
    """
    # Align and drop NaN
    mask = actual.notna() & predicted.notna()
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) < 2:
        return {
            "mae": None,
            "rmse": None,
            "mape": None,
            "bias": None,
            "r2": None,
            "count": len(actual),
            "total_actual": None,
            "total_predicted": None,
        }
    
    errors = predicted - actual
    abs_errors = np.abs(errors)
    
    # MAE: Mean Absolute Error
    mae = abs_errors.mean()
    
    # RMSE: Root Mean Square Error
    rmse = np.sqrt((errors ** 2).mean())
    
    # MAPE: Mean Absolute Percentage Error (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_errors = abs_errors / actual * 100
        pct_errors = pct_errors.replace([np.inf, -np.inf], np.nan)
        mape = pct_errors.mean()
    
    # Bias: Mean Error (positive = over-prediction)
    bias = errors.mean()
    
    # R²: Coefficient of determination
    ss_res = (errors ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "bias": bias,
        "r2": r2,
        "count": len(actual),
        "total_actual": actual.sum(),
        "total_predicted": predicted.sum(),
    }


def check_within_target(actual_df, predicted_df):
    """
    Check how many predictions fall within Meteocontrol target range.
    
    Returns:
        dict with within_count, total_count, percentage
    """
    aligned = actual_df.join(predicted_df[["predicted_kwh"]], how="inner")
    aligned = aligned.dropna(subset=["predicted_kwh", "target_low", "target_high"])
    
    if len(aligned) == 0:
        return {"within_count": 0, "total_count": 0, "percentage": 0}
    
    within = (
        (aligned["predicted_kwh"] >= aligned["target_low"]) & 
        (aligned["predicted_kwh"] <= aligned["target_high"])
    )
    
    return {
        "within_count": within.sum(),
        "total_count": len(aligned),
        "percentage": within.mean() * 100
    }


# =====================================================================
# OUTPUT FUNCTIONS
# =====================================================================

def print_summary_table(results, granularity="daily"):
    """Print formatted terminal table with results."""
    # Determine unit and count label based on granularity
    if granularity == "hourly":
        unit = "kW"
        count_label = "Hours"
    else:
        unit = "kWh"
        count_label = "Days" if granularity == "daily" else "Months"
    
    print("\n" + "=" * 78)
    print(f"                    {granularity.upper()} METRICS")
    print("=" * 78)
    print(f"{'Source':<12} │ {'MAE('+unit+')':>9} │ {'RMSE('+unit+')':>10} │ {'MAPE(%)':>8} │ {'Bias':>7} │ {'R²':>5} │ {count_label:>6}")
    print("-" * 78)
    
    for source, metrics in results.items():
        if metrics["mae"] is not None:
            mae_str = f"{metrics['mae']:.1f}"
            rmse_str = f"{metrics['rmse']:.1f}"
            mape_str = f"{metrics['mape']:.1f}%"
            bias_str = f"{metrics['bias']:+.1f}"
            r2_str = f"{metrics['r2']:.2f}"
            count_str = str(metrics["count"])
        else:
            mae_str = rmse_str = mape_str = bias_str = r2_str = "N/A"
            count_str = "0"
        
        print(f"{source:<12} │ {mae_str:>9} │ {rmse_str:>10} │ {mape_str:>8} │ {bias_str:>7} │ {r2_str:>5} │ {count_str:>6}")
    
    print("=" * 78)
    
    # Print energy totals
    print("\nEnergy Totals:")
    for source, metrics in results.items():
        if metrics["total_actual"] is not None:
            print(f"  {source}: Predicted {metrics['total_predicted']:.1f} {unit} vs Actual {metrics['total_actual']:.1f} {unit}")


def save_combined_report(all_results, actual_df, all_predictions, output_path):
    """Save combined multi-granularity text report to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("              MULTI-GRANULARITY ACCURACY COMPARISON REPORT\n")
        f.write(f"                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 78 + "\n\n")
        
        # Write metrics for each granularity
        for granularity, results in all_results.items():
            # Determine unit and count label based on granularity
            if granularity == "hourly":
                unit = "kW"
                count_label = "Hours"
            else:
                unit = "kWh"
                count_label = "Days" if granularity == "daily" else "Months"
            
            f.write(f"{granularity.upper()} METRICS\n")
            f.write("-" * 78 + "\n")
            f.write(f"{'Source':<12} │ {'MAE('+unit+')':>9} │ {'RMSE('+unit+')':>10} │ {'MAPE(%)':>8} │ {'Bias':>7} │ {'R²':>5} │ {count_label:>6}\n")
            f.write("-" * 78 + "\n")
            
            for source, metrics in results.items():
                if metrics["mae"] is not None:
                    f.write(f"{source:<12} │ {metrics['mae']:>9.1f} │ {metrics['rmse']:>10.1f} │ {metrics['mape']:>7.1f}% │ {metrics['bias']:>+7.1f} │ {metrics['r2']:>5.2f} │ {metrics['count']:>6}\n")
                else:
                    f.write(f"{source:<12} │ {'N/A':>9} │ {'N/A':>10} │ {'N/A':>8} │ {'N/A':>7} │ {'N/A':>5} │ {'0':>6}\n")
            
            f.write("\n")
        
        # Daily details (if daily predictions available)
        if "daily" in all_predictions:
            predictions_dict = all_predictions["daily"]
            f.write("DAILY COMPARISON DETAILS\n")
            f.write("-" * 78 + "\n")
            f.write(f"{'Day':>3} │ {'Actual':>8} │")
            for source in all_results.get("daily", {}).keys():
                f.write(f" {source[:10]:>10} │")
            f.write("\n")
            f.write("-" * 78 + "\n")
            
            for day in sorted(actual_df.index):
                actual_val = actual_df.loc[day, "actual_kwh"]
                f.write(f"{day:>3} │ {actual_val:>8.1f} │")
                
                for source, pred_df in predictions_dict.items():
                    if pred_df is not None and day in pred_df.index:
                        pred_val = pred_df.loc[day, "predicted_kwh"]
                        f.write(f" {pred_val:>10.1f} │")
                    else:
                        f.write(f" {'---':>10} │")
                f.write("\n")
            
            f.write("\n")
        
        f.write("=" * 78 + "\n")
        f.write("END OF REPORT\n")
    
    print(f"✓ Combined report saved: {output_path}")


def save_json_report(all_results, output_path):
    """Save results as JSON."""
    # Convert numpy types to Python types
    json_results = {}
    for granularity, results in all_results.items():
        json_results[granularity] = {}
        for source, metrics in results.items():
            json_results[granularity][source] = {}
            for k, v in metrics.items():
                if isinstance(v, (np.floating, np.integer)):
                    json_results[granularity][source][k] = float(v) if v is not None else None
                elif isinstance(v, dict):
                    # Handle nested dicts like within_target
                    json_results[granularity][source][k] = {
                        kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                        for kk, vv in v.items()
                    }
                else:
                    json_results[granularity][source][k] = v
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✓ JSON report saved: {output_path}")


# =====================================================================
# VISUALIZATION
# =====================================================================

def generate_comparison_plot(results, output_path):
    """Create bar chart comparing accuracy metrics across sources."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping plots")
        return
    
    # Filter sources with valid data
    valid_sources = {k: v for k, v in results.items() if v["mae"] is not None}
    
    if not valid_sources:
        print("⚠️ No valid data for comparison plot")
        return
    
    sources = list(valid_sources.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Accuracy Comparison by Data Source", fontsize=14, fontweight="bold")
    
    # MAE
    ax1 = axes[0, 0]
    mae_vals = [valid_sources[s]["mae"] for s in sources]
    bars1 = ax1.bar(sources, mae_vals, color=["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"])
    ax1.set_ylabel("MAE (kWh)")
    ax1.set_title("Mean Absolute Error (lower is better)")
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, mae_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # RMSE
    ax2 = axes[0, 1]
    rmse_vals = [valid_sources[s]["rmse"] for s in sources]
    bars2 = ax2.bar(sources, rmse_vals, color=["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"])
    ax2.set_ylabel("RMSE (kWh)")
    ax2.set_title("Root Mean Square Error (lower is better)")
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # MAPE
    ax3 = axes[1, 0]
    mape_vals = [valid_sources[s]["mape"] for s in sources]
    bars3 = ax3.bar(sources, mape_vals, color=["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"])
    ax3.set_ylabel("MAPE (%)")
    ax3.set_title("Mean Absolute Percentage Error (lower is better)")
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, mape_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%', 
                ha='center', va='bottom', fontsize=9)
    
    # R²
    ax4 = axes[1, 1]
    r2_vals = [valid_sources[s]["r2"] for s in sources]
    bars4 = ax4.bar(sources, r2_vals, color=["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"])
    ax4.set_ylabel("R²")
    ax4.set_title("Coefficient of Determination (higher is better)")
    ax4.set_ylim(0, 1.1)
    ax4.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars4, r2_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved: {output_path}")


def generate_daily_plot(actual_df, predictions_dict, output_path):
    """Create line chart showing actual vs all predictions by day."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib not installed, skipping plots")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    days = sorted(actual_df.index)
    
    # Plot actual values
    ax.plot(days, actual_df.loc[days, "actual_kwh"], 
            'ko-', linewidth=2, markersize=6, label="Actual (Meteocontrol)")
    
    # Plot target range as shaded area
    if "target_low" in actual_df.columns and "target_high" in actual_df.columns:
        target_low = actual_df.loc[days, "target_low"].values
        target_high = actual_df.loc[days, "target_high"].values
        ax.fill_between(days, target_low, target_high, alpha=0.2, color="gray", label="Target Range")
    
    # Plot predictions
    colors = {"NASA POWER": "#2ecc71", "Solcast": "#3498db", "Open-Meteo": "#e74c3c", "ERA5": "#9b59b6"}
    markers = {"NASA POWER": "s", "Solcast": "^", "Open-Meteo": "d", "ERA5": "v"}
    
    for source, pred_df in predictions_dict.items():
        if pred_df is None:
            continue
        
        # Get predictions for available days
        pred_days = [d for d in days if d in pred_df.index]
        pred_vals = [pred_df.loc[d, "predicted_kwh"] for d in pred_days]
        
        if pred_days:
            ax.plot(pred_days, pred_vals, 
                   marker=markers.get(source, "o"), 
                   color=colors.get(source, "gray"),
                   linestyle="--", linewidth=1.5, markersize=5,
                   label=source, alpha=0.8)
    
    ax.set_xlabel("Day of Month")
    ax.set_ylabel("Daily Energy (kWh)")
    ax.set_title("Daily Generation: Actual vs Predictions")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    ax.set_xticks(days)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Daily plot saved: {output_path}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare predictions against Meteocontrol actual data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_accuracy.py                    # Default comparison
  python compare_accuracy.py --actual custom.csv
  python compare_accuracy.py --days 1-15
  python compare_accuracy.py --no-plot
  python compare_accuracy.py --json
        """
    )
    
    parser.add_argument(
        "--actual",
        default=str(DEFAULT_METEOCONTROL),
        help="Path to Meteocontrol actual data CSV"
    )
    parser.add_argument(
        "--days",
        help="Day range to compare (e.g., '1-15' or '5-20')"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip graphical output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--granularity",
        choices=["hourly", "daily", "monthly", "all"],
        default="all",
        help="Granularity to compare (default: all)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("           ACCURACY COMPARISON")
    print("=" * 60)
    
    # Load Meteocontrol actual data
    actual_df = load_meteocontrol(Path(args.actual))
    
    # Filter by day range if specified
    if args.days:
        try:
            start, end = map(int, args.days.split("-"))
            actual_df = actual_df[(actual_df.index >= start) & (actual_df.index <= end)]
            print(f"Filtered to days {start}-{end}: {len(actual_df)} days")
        except ValueError:
            print(f"⚠️ Invalid day range format: {args.days}. Use format like '1-15'")
    
    # Determine which granularities to process
    if args.granularity == "all":
        granularities = ["hourly", "daily", "monthly"]
    else:
        granularities = [args.granularity]
    
    # Store results for all granularities
    all_results = {}
    all_predictions = {}
    
    # Process each granularity
    for granularity in granularities:
        print(f"\n{'='*60}")
        print(f"  {granularity.upper()} ANALYSIS")
        print(f"{'='*60}")
        
        # Get prediction files for this granularity
        prediction_files = get_prediction_files(granularity)
        
        # Load predictions from all sources
        print("\nLoading predictions:")
        predictions_dict = {}
        for source, path in prediction_files.items():
            predictions_dict[source] = load_predictions(path, source, granularity)
        
        # Calculate metrics for each source
        print("\nCalculating metrics...")
        results = {}
        
        for source, pred_df in predictions_dict.items():
            if pred_df is None:
                results[source] = {
                    "mae": None, "rmse": None, "mape": None, 
                    "bias": None, "r2": None, "count": 0,
                    "total_actual": None, "total_predicted": None
                }
                continue
            
            # Align with actual data
            aligned = actual_df.join(pred_df[["predicted_kwh"]], how="inner")
            aligned = aligned.dropna(subset=["actual_kwh", "predicted_kwh"])
            
            if len(aligned) == 0:
                results[source] = {
                    "mae": None, "rmse": None, "mape": None, 
                    "bias": None, "r2": None, "count": 0,
                    "total_actual": None, "total_predicted": None
                }
                continue
            
            metrics = calculate_metrics(aligned["actual_kwh"], aligned["predicted_kwh"])
            
            # Add within-target info
            target_info = check_within_target(actual_df, pred_df)
            metrics["within_target"] = target_info
            
            results[source] = metrics
        
        # Store results for this granularity
        all_results[granularity] = results
        all_predictions[granularity] = predictions_dict
        
        # Print summary table for this granularity
        print_summary_table(results, granularity)
    
    # Save combined text report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_combined_report(all_results, actual_df, all_predictions, OUTPUT_DIR / "accuracy_report.txt")
    
    # Save JSON if requested
    if args.json:
        save_json_report(all_results, OUTPUT_DIR / "accuracy_report.json")
    
    # Generate plots (use daily results for plots)
    if not args.no_plot and "daily" in all_results:
        print("\nGenerating plots...")
        generate_comparison_plot(all_results["daily"], OUTPUT_DIR / "accuracy_comparison.png")
        if "daily" in all_predictions:
            generate_daily_plot(actual_df, all_predictions["daily"], OUTPUT_DIR / "daily_comparison.png")
    
    print("\n" + "=" * 60)
    print("           COMPARISON COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()


import pandas as pd

def create_predictive_conflict_labels(df, prediction_horizon=5):
    # Sort by robot and time to ensure sequential alignment
    print("\n[1a] Sorting data chronologically per robot...")
    df = df.sort_values(by=["robot_id", "t"]).reset_index(drop=True)
    
    # Check if the robot 'is_deadlocked' or 'is_stuck' within the next N steps
    # We use a rolling maximum looking forward (using a backward roll on reversed data)
    print("\n[1b] Creating predictive conflict label via forward-lookahead...")
    df['is_conflict'] = (
        df['is_stuck'].astype(bool) | 
        df['is_deadlocked'].astype(bool) | 
        (df['collision_vertex'] > 0) | 
        (df['collision_edge'] > 0)
    ).astype(int)
    
    print("\n[1c] Shifting to create future conflict label...")
    df['target_future_conflict'] = df.groupby('robot_id')['is_conflict'].shift(-prediction_horizon)
    df = df.dropna(subset=['target_future_conflict'])  # Drop rows where future label is undefined
    return df

def encode_categoricals(df):
    # One-hot encode the robot states
    print("\n[2a] One-hot encoding robot states...")
    df = pd.get_dummies(df, columns=["state"], prefix="state", dtype=int)
    
    # Binary encode if the robot is carrying an item
    print("\n[2b] Encoding if robot is carrying an item...")
    df["is_carrying"] = df["held_item"].notna().astype(int)
    df = df.drop(columns=["held_item"]) # Drop original string column
    
    return df

def engineer_spatial_features(df):
    # Find the warehouse dimensions to normalize coordinates
    print("\n[3a] Normalizing spatial coordinates...")
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()
    
    # Normalise x and y coordinates
    df["x_norm"] = (df["x"] - x_min) / (x_max - x_min) if x_max != x_min else 0
    df["y_norm"] = (df["y"] - y_min) / (y_max - y_min) if y_max != y_min else 0

    # NOTE: vertex_risk_factor is intentionally NOT computed here.
    # It must be derived only from the training partition (after the train/test split)
    # to avoid target leakage. It is computed in the evaluation script instead.

    return df

def create_rolling_features(df, window_size=5):
    df = df.sort_values(by=["robot_id", "t"]).reset_index(drop=True)
    
    # Calculate the change in density over the last 5 time steps
    print("\n[4a] Creating rolling features for local density 3...")
    df['local_density_3_smooth'] = (
        df.groupby('robot_id')['local_density_3']
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    )
    
    print("\n[4b] Creating rolling features for local density 5...")
    df['local_density_5_smooth'] = (
        df.groupby('robot_id')['local_density_5']
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    )

    # Capture movement velocity from raw coordinates
    print("\n[4c] Calculating movement velocity...")
    df["dx"] = df.groupby("robot_id")["x"].diff().fillna(0)
    df["dy"] = df.groupby("robot_id")["y"].diff().fillna(0)
    df["speed"] = (df["dx"]**2 + df["dy"]**2)**0.5
    
    return df

def normalize_density_features(df, window_size=5):
    """
    Adds scale-invariant density features normalised per robot's recent history.
    These transfer better across scenarios with different baseline density levels,
    directly addressing the cross-dataset generalisation problem.

    Features added:
      - density_3_zscore : (d3 - rolling_mean) / (rolling_std + 0.1)
                           Relative congestion vs this robot's recent baseline.
      - density_5_zscore : same for wider radius
      - density_3_peak   : rolling max over window — captures spike events
                           that the rolling mean smooths away.
      - density_5_peak   : same for wider radius
      - density_3_accel  : second difference — how fast is congestion
                           accelerating? Detects build-up earlier than delta.
      - density_5_accel  : same for wider radius
    """
    print("\n[5a] Computing per-robot density z-scores (scale-invariant)...")
    df = df.sort_values(by=["robot_id", "t"]).reset_index(drop=True)

    for col, prefix in [
        ("local_density_3", "density_3"),
        ("local_density_5", "density_5"),
    ]:
        grp = df.groupby("robot_id")[col]

        roll_mean = grp.transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        roll_std = grp.transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std().fillna(0)
        )

        # Z-score: how many std-devs above/below this robot's recent average?
        # Clipped to [-5, 5] to prevent outlier explosion.
        df[f"{prefix}_zscore"] = (
            (df[col] - roll_mean) / (roll_std + 0.1)
        ).clip(-5, 5)

        # Rolling peak: worst-case density seen in the last window steps
        df[f"{prefix}_peak"] = grp.transform(
            lambda x: x.rolling(window=window_size, min_periods=1).max()
        )

        # Acceleration: second difference — rate of change of the rate of change
        # Positive = congestion building faster; negative = dispersing faster
        df[f"{prefix}_accel"] = grp.transform(
            lambda x: x.diff(window_size).diff(window_size).fillna(0)
        )

    print("[5b] Done: z-score, peak, acceleration computed for density_3 and density_5.")
    return df

def create_behavioral_features(df, window_size=5, num_lags=3):
    """
    Engineers genuine leading-indicator features that capture robot dynamics
    BEFORE a deadlock occurs — no current conflict state is encoded.

    Features added:
      - steps_stationary      : consecutive steps at the same (x,y) grid cell
                                (robot slowing/stopping before deadlock)
      - stuck_duration_lag_N  : stuck_duration N steps ago — a mild early
                                warning signal without encoding current state
      - density_3_delta       : change in local density over last window_size
                                steps (is congestion building up?)
      - density_5_delta       : same for the wider radius
      - density_spread        : density_5 / density_3 ratio — are nearby
                                robots concentrated (bottleneck) or spread out?
      - density_3_lag_N       : lagged snapshots of local crowding (N=1..num_lags)
      - density_5_lag_N       : lagged snapshots of wider crowding
    """
    print("\n[6a] Computing position-stationarity (steps at same grid cell)...")
    df = df.sort_values(by=["robot_id", "t"]).reset_index(drop=True)

    # Position key per row
    df["_pos_key"] = df["x"].astype(str) + "_" + df["y"].astype(str)

    # 1 if position changed from previous step, 0 if robot stayed in same cell
    df["_pos_changed"] = (
        df.groupby("robot_id")["_pos_key"].shift(1) != df["_pos_key"]
    ).astype(int).fillna(1)

    # Cumsum of changes creates block IDs; cumcount within each block = steps stationary
    df["_block"] = df.groupby("robot_id")["_pos_changed"].cumsum()
    df["steps_stationary"] = df.groupby(["robot_id", "_block"]).cumcount()

    # Stationary ratio: bounded [0.0–1.0] fraction of last N steps without moving.
    # More interpretable than raw cumulative count; doesn't grow unboundedly.
    df["_not_changed"] = 1 - df["_pos_changed"]
    df["stationary_ratio"] = df.groupby("robot_id")["_not_changed"].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )

    df.drop(columns=["_pos_key", "_pos_changed", "_not_changed", "_block"], inplace=True)

    # Neighbour stationarity count: how many OTHER robots are frozen at the same t?
    # System-wide signal, completely scale-independent across scenarios.
    print("\n[6b] Computing neighbour stationarity count (system-wide pressure)...")
    static_at_t = df.groupby("t")["steps_stationary"].transform(lambda x: (x > 0).sum())
    df["neighbour_static_count"] = (
        static_at_t - (df["steps_stationary"] > 0).astype(int)
    ).clip(lower=0)

    # NOTE: stuck_duration_lag_N was removed — even lagged by 1 step,
    # stuck_duration encodes near-current stuck state (deadlocks persist
    # across steps), causing the model to shortcut instead of learning
    # genuine leading behavioural patterns.

    print("\n[6c] Computing density trend (rate of congestion change)...")
    # Positive delta = congestion growing; negative = robots dispersing
    df["density_3_delta"] = (
        df.groupby("robot_id")["local_density_3"]
        .transform(lambda x: x.diff(periods=window_size).fillna(0))
    )
    df["density_5_delta"] = (
        df.groupby("robot_id")["local_density_5"]
        .transform(lambda x: x.diff(periods=window_size).fillna(0))
    )

    print("\n[6d] Computing density spread (bottleneck vs wide congestion)...")

    # High spread (density_5 >> density_3) = many robots nearby but spread out
    # Low spread (density_5 ≈ density_3) = tight cluster — deadlock risk
    df["density_spread"] = (
        df["local_density_5"] / df["local_density_3"].replace(0, 1)
    ).clip(upper=10)  # cap outliers from near-zero density_3

    print(f"\n[6e] Creating density lag features (N=1..{num_lags})...")

    for lag in range(1, num_lags + 1):
        df[f"density_3_lag_{lag}"] = (
            df.groupby("robot_id")["local_density_3"].shift(lag).fillna(0)
        )
        df[f"density_5_lag_{lag}"] = (
            df.groupby("robot_id")["local_density_5"].shift(lag).fillna(0)
        )

    return df

def preprocess_congestion_dataset(df, lead_time_steps=5, window_size=5, num_lags=3):

    print("\n[1/6] Creating predictive conflict labels...")
    df = create_predictive_conflict_labels(df, prediction_horizon=lead_time_steps)

    print("\n[2/6] Encoding categorical features...")
    df = encode_categoricals(df)

    print("\n[3/6] Engineering spatial features...")
    df = engineer_spatial_features(df)

    print("\n[4/6] Creating rolling/velocity features...")
    df = create_rolling_features(df, window_size=window_size)

    print("\n[5/6] Normalising density features (z-score, peak, acceleration)...")
    df = normalize_density_features(df, window_size=window_size)

    print("\n[6/6] Engineering behavioral leading-indicator features...")
    df = create_behavioral_features(df, window_size=window_size, num_lags=num_lags)

    columns_to_drop = [
        # ── Current conflict state indicators (target components) ──────────
        # These directly encode the current conflict state and would leak the
        # answer. stuck_duration_lag_N variants are kept as leading indicators.
        'is_conflict', 'is_stuck', 'is_deadlocked', 'stuck_duration',
        'collision_vertex', 'collision_edge',
        # ── Current-state-only columns ─────────────────────────────────────
        # path_length is ONLY non-zero when already deadlocked.
        'path_length',
        # dx, dy, speed are all-zero in this simulator (robots teleport).
        'dx', 'dy', 'speed',
        # ── Raw columns replaced by engineered versions ────────────────────
        # Raw coordinates → x_norm, y_norm
        'x', 'y',
        # Raw density → smoothed + delta + lag variants
        'local_density_3', 'local_density_5',
    ]
    df = df.drop(columns=columns_to_drop)
    
    # Drop rows at the end of trajectories where the shifted future target is NaN
    df = df.dropna(subset=['target_future_conflict']).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Example usage
    print("\n[INFO] Starting dataset preprocessing...")
    source_location = "congestion_dataset.csv"  # Path to your raw dataset
    processed_location = "processed_congestion_dataset.csv"  # Path to save the processed dataset
    
    print(f"\n[INFO] Loading dataset from {source_location}...")
    df = pd.read_csv(source_location)  # Load your dataset
    df = preprocess_congestion_dataset(df, lead_time_steps=5, window_size=5)
    
    print(f"\n[INFO] Saving processed dataset to {processed_location}...")
    df.to_csv(processed_location, index=False)  # Save the processed dataset
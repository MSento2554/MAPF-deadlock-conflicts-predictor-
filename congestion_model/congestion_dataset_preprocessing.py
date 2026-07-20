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

    print("\n[3b] Calculating vertex risk factors...")
    vertex_mentions = df['collision_vertex'].value_counts()
    
    # Number of times a collision/stuck state actually occurred on that vertex
    # (Assuming a collision is recorded when the robot is currently stuck/deadlocked)
    vertex_collisions = df[df['is_conflict'] == True]['collision_vertex'].value_counts()
    
    # Calculate risk factor: (collisions / mentions) and fill unproblematic vertices with 0
    risk_factor_map = (vertex_collisions / vertex_mentions).fillna(0)
    
    # Map back to the dataframe (automatically normalizes between 0.0 and 1.0 based on ratio)
    df['vertex_risk_factor'] = df['collision_vertex'].map(risk_factor_map)
    
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

def preprocess_congestion_dataset(df, lead_time_steps=5, window_size=5):

    print("\n[1/4] Creating predictive conflict labels...")
    df = create_predictive_conflict_labels(df, prediction_horizon=lead_time_steps)

    print("\n[2/4] Encoding categorical features...")
    df = encode_categoricals(df)

    print("\n[3/4] Engineering spatial features...")
    df = engineer_spatial_features(df)

    print("\n[4/4] Creating rolling features...")
    df = create_rolling_features(df, window_size=window_size)

    columns_to_drop = [
        'is_conflict', 'is_stuck', 'is_deadlocked', 'stuck_duration', 
        'x', 'y', 'local_density_3', 'local_density_5'
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
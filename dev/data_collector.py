import sys
import os

# Remove the script's directory from sys.path to avoid shadowing the global 'redis' package
# by the local './redis/' directory in dev/
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

import json
import csv
import time
import redis

def main():
    REDIS_HOST = os.getenv("REDIS_HOST", default="localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", default="6379"))
    
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    try:
        redis_con = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        redis_con.ping()
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        print("Please ensure Docker Compose or your local Redis service is running.")
        return

    csv_file = "congestion_dataset.csv"
    csv_headers = [
        "t",
        "robot_id",
        "x",
        "y",
        "state",
        "held_item",
        "path_length",
        "local_density_3",
        "local_density_5",
        "stuck_duration",
        "collision_vertex",
        "collision_edge",
        "is_stuck",
        "is_deadlocked"
    ]
    
    # Check if we should append or write headers
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode="a" if file_exists else "w", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(csv_headers)
            
    print(f"Logging data to {csv_file}")
    
    # Track historical data for robots: robot_id -> {"last_pos": (x, y), "stuck_duration": count}
    robot_history = {}
    
    # Read stream from the current end
    last_id = '$'
    
    # First, try reading existing events in the stream buffer by starting from '0'
    try:
        existing = redis_con.xread({'world:state': '0'}, count=100)
        if existing:
            print(f"Found {len(existing[0][1])} existing state events in Redis stream. Processing historical events...")
            last_id = '0'
    except Exception:
        last_id = '$'

    print("Monitoring world state stream...")
    try:
        while True:
            response = redis_con.xread({'world:state': last_id}, block=2000, count=10)
            if not response:
                time.sleep(0.1)
                continue
                
            for stream_name, events in response:
                for event_id, data in events:
                    last_id = event_id # update last read ID
                    
                    t = int(data['t'])
                    robots_data = json.loads(data['robots'])
                    
                    # 1. Parse robot states
                    parsed_robots = []
                    for r_data in robots_data:
                        r_id = int(r_data['robot_id'])
                        pos = tuple(json.loads(r_data['position']))
                        state = r_data['state']
                        held_item = 1 if r_data['held_item_id'] != '' else 0
                        path_len = len(r_data['path'])
                        parsed_robots.append({
                            'id': r_id,
                            'pos': pos,
                            'state': state,
                            'held_item': held_item,
                            'path_len': path_len
                        })
                    
                    # 2. Extract features and labels for each robot
                    rows_to_write = []
                    for r in parsed_robots:
                        r_id = r['id']
                        pos = r['pos']
                        
                        # Get previous state history
                        history = robot_history.get(r_id, {'last_pos': pos, 'stuck_duration': 0})
                        prev_pos = history['last_pos']
                        stuck_duration = history['stuck_duration']
                        
                        # Calculate stuck duration:
                        # Only increment if the robot has a path (wants to move) but did not change position
                        if r['path_len'] > 0 and pos == prev_pos:
                            stuck_duration += 1
                        elif pos != prev_pos:
                            stuck_duration = 0
                            
                        # Update history
                        robot_history[r_id] = {
                            'last_pos': pos,
                            'stuck_duration': stuck_duration
                        }
                        
                        # Calculate local densities (other robots in Manhattan distance of 3 and 5)
                        density_3 = 0
                        density_5 = 0
                        collision_vertex = 0
                        collision_edge = 0
                        
                        for other in parsed_robots:
                            if other['id'] == r_id:
                                continue
                            
                            other_pos = other['pos']
                            dist = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                            
                            if dist <= 3:
                                density_3 += 1
                            if dist <= 5:
                                density_5 += 1
                            
                            # Vertex collision check (same position)
                            if dist == 0:
                                collision_vertex = 1
                                
                            # Edge collision check (swapped positions)
                            other_history = robot_history.get(other['id'])
                            if other_history:
                                other_prev_pos = other_history['last_pos']
                                if prev_pos == other_pos and pos == other_prev_pos:
                                    collision_edge = 1
                                    
                        is_stuck = 1 if stuck_duration >= 2 else 0
                        is_deadlocked = 1 if stuck_duration >= 6 else 0
                        
                        rows_to_write.append([
                            t, r_id, pos[0], pos[1], r['state'], r['held_item'], r['path_len'],
                            density_3, density_5, stuck_duration, collision_vertex, collision_edge,
                            is_stuck, is_deadlocked
                        ])
                    
                    # Write to CSV
                    with open(csv_file, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(rows_to_write)
                        
            print(f"Step {t} logged to {csv_file}")
            
    except KeyboardInterrupt:
        print("\nData collection stopped.")

if __name__ == "__main__":
    main()

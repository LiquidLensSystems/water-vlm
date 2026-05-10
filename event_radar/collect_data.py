import requests
import os
import time
import numpy as np
from PIL import Image
import math
import reverse_geocode 
import json
import base64
import geopandas as gpd
from global_land_mask import globe
from openai import OpenAI
import base64
import cv2



# =========================================================
# CONFIG
# =========================================================

BASE_URL = "http://localhost:9005"
DASHBOARD = "http://localhost:8000"
SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

client = OpenAI(
    base_url="http://localhost:8080/v1",  # The hosted llama-server
    api_key="not-needed"
)
 
# =========================================================
# ASSETS
# =========================================================

lakes_path = os.path.join("data", "ne_10m_lakes", "ne_10m_lakes.shp")

lakes = gpd.read_file(lakes_path)

# Convert to asset format
ASSETS = []
for idx, row in lakes.iterrows():
    # skip invalid geometry
    if row.geometry is None:
        continue
    centroid = row.geometry.centroid
    lat = centroid.y
    lon = centroid.x

    # skip invalid coordinates
    if math.isnan(lat) or math.isnan(lon):
        continue

    # fallback name
    name = row.get("name")

    if name is None or str(name) == "nan":
        name = f"Lake_{idx}"

    ASSETS.append({
        "name": name,
        "lat": centroid.y,
        "lon": centroid.x
    })

asset_state = {
    asset["name"]: {
        "lat": asset["lat"],
        "lon": asset["lon"],
        "status": "IDLE",
        "color": "gray"
    }
    for asset in ASSETS
}
with open("mission_control/shared_state.json", "w") as f:
    json.dump({"assets": [dict(name=n, **d) for n, d in asset_state.items()]}, f)

# =========================================================
# REALISM SETTINGS
# =========================================================
TRIGGER_RADIUS_KM = 200  # Only take a photo if within this radius

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the distance in KM between two points on Earth."""
    R = 6371 
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2)**2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
         math.sin(d_lon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# =========================================================
# SIMULATION TIME
# =========================================================

def get_position():
    r = requests.get(f"{BASE_URL}/data/current/position", timeout=10)
    return r.json() # Returns full dict: {"lat": x, "lon": y, "timestamp": z}

def safe_timestamp(ts):
    return ts.replace(":", "-")

# =========================================================
# IMAGE FETCH
# =========================================================

def fetch_image(lat, lon, timestamp):
    return requests.get(
        f"{BASE_URL}/data/image/sentinel",
        params={"lat": lat, "lon": lon, "timestamp": timestamp, "return_type": "png"},
        timeout=30
    )

def analyze_water_risk(resp_json):
    try:
        meta = resp_json["image"]["metadata"]
        raw_b64 = resp_json["image"]["image"]

        # Decode base64
        raw = base64.b64decode(raw_b64)

        # Convert to numpy
        arr = np.frombuffer(raw, dtype=np.uint16)

        # Reshape
        shape = meta["shape"]  # (bands, H, W)
        arr = arr.reshape(shape)

        # Convert to (H, W, bands)
        arr = np.transpose(arr, (1, 2, 0))

        bands = meta["bands"]

        green = arr[:, :, bands.index("green")].astype(np.float32)
        red   = arr[:, :, bands.index("red")].astype(np.float32)
        nir   = arr[:, :, bands.index("nir")].astype(np.float32)
        red_edge = arr[:, :, bands.index("rededge1")].astype(np.float32)
        
        eps = 1e-6
        # NDWI (Water Index)
        ndwi = (green - nir) / (green + nir + eps)
        water_mask = ndwi > 0.1
        water_score = np.mean(water_mask)
        
        # SUSPENDED SOLIDS (Turbidity) Range: -1 to 1. Higher = Muddier.
        ndti = (red - green) / (red + green + eps)        
        turbidity_score = np.mean(ndti)

        # ALGAE PRESENCE (Chlorophyll) Range: -1 to 1. Higher = More Algae.
        ndci = (red_edge - red) / (red_edge + red + 1e-5)
        algae_index = np.mean(ndci)

        # DEPTH ESTIMATION (Relative) Higher nir_absorb = Deeper or clearer water.
        nir_absorb = 1.0 - (np.mean(nir) / 65535.0)

        # INITIALIZE DEFAULTS
        status = "CLEAR WATER"
        color = "lime"
        trigger_vlm = False

        # IF-ELSE PRIORITY LOOP
        # Priority 1: Check for High Sediment (RED ALERT)
        if turbidity_score > 0.05:
            status = f"HIGH TURBIDITY / SEDIMENT {round(float(turbidity_score), 3)}"
            color = "red"
            trigger_vlm = True # VLM needed to identify source
            # Check if algae is also present to append to status
            if algae_index > 0.1:
                status += f" & ALGAE {round(float(algae_index), 3)}"

        # Priority 2: Check for Algae (ORANGE ALERT)
        elif algae_index > 0.1:
            status = f"ALGAE BLOOM DETECTED {round(float(algae_index), 3)}"
            color = "orange"
            trigger_vlm = True # VLM needed to check for toxic scum

        # Priority 3: Drought / Extreme Shallows
        elif nir_absorb < 0.85 and water_score > 0.05:
            status = f"WARNING: EXTREME SHALLOWS / DROUGHT {round(float(nir_absorb), 3)}"
            color = "orange"
            trigger_vlm = True # VLM needed to verify if the bed is exposed

        # Priority 4: Check for missing water (Drought/Low Level)
        elif water_score < 0.05:
            status = f"LOW WATER LEVEL {round(float(water_score), 3)}"
            color = "yellow" # Yellow for warning/low volume
            trigger_vlm = False
        
        elif status == "CLEAR WATER" and nir_absorb > 0.98:
            status = "HEALTHY DEEP WATER"

        metrics = {
            "water_score": float(water_score),
            "turbidity": float(turbidity_score),
            "algae_index": float(algae_index),
            "nir_absorb": float(nir_absorb)
        }

        return metrics, status, color, trigger_vlm, arr, bands

    except Exception as e:
        return f"ANALYSIS ERROR: {str(e)}", "gray", False, None, None


def call_liquid_vlm(metrics, risk_label, arr, bands):
    """Refines spectral math with visual intelligence using the passed NumPy array."""
    
    # Map Multispectral bands to RGB (Red-Green-Blue)
    # Satellites use 'uint16', but VLM needs 'uint8' (0-255)
    r = arr[:, :, bands.index("red")]
    g = arr[:, :, bands.index("green")]
    # SimSat doesn't have blue, so we use green or a mix as a proxy for the 'Blue' channel
    b = arr[:, :, bands.index("green")] 

    # Stack to create a 3-channel BGR image for OpenCV
    rgb_img = np.stack([b, g, r], axis=-1)
    
    # Normalize to 8-bit so it's not a black image (scales the 16-bit satellite data (0-65535) to 8-bit (0-255))
    rgb_8bit = cv2.normalize(rgb_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rgb_8bit = cv2.resize(rgb_8bit, (448, 448))

    # Convert the NumPy array directly to Base64
    _, buffer = cv2.imencode(".jpg", rgb_8bit)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    prompt = (
            f"ANOMALY: {metrics}. IMAGE SPECTRAL BANDS: {bands}. TASK: Validate visually. "
            "Briefly describe only the most distinct textures or shoreline features. "
            "Focus on evidence of sediment, algae mats, or exposed beds. "
            "Limit response to 2-3 technical sentences. No introductory filler."
        )

    # API Call to local llama-server
    response = client.chat.completions.create(
        model="LFM2.5-VL-450M-new-chat-template-3", 
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }],
        temperature=0.1,
        max_tokens=256,
        extra_body={"min_p": 0.15, "repetition_penalty": 1.05},
    )
    
    return response.choices[0].message.content

# =========================================================
# MISSION INITIALIZATION (Orchestration Layer)
# =========================================================

def initialize_mission():
    print("Initializing Mission Control...")
    try:
        # Stop any existing simulation to reset the clock
        requests.post(f"{DASHBOARD}/api/commands/", json={"command": "stop"})
        time.sleep(1)
        
        # Start the simulation with a high replay_speed so we don't wait forever for flyovers
        setup_params = {
            "command": "start",
            "start_time": "2023-04-01T08:30:00Z", # Desired start
            "step_size_seconds": 120,             # Jump by steps size on every 'tick'
            "replay_speed": 50                   # e.g. 5x real-time speed
        }
        
        resp = requests.post(f"{DASHBOARD}/api/commands/", json=setup_params)
        if resp.status_code in [200, 201]:
            print(f"✅ Mission Synchronized! ID: {resp.json().get('id')}")
            print(f"📅 Start Time Set to: {setup_params['start_time']}")
        else:
            print(f"❌ Command Rejected: {resp.status_code} - {resp.text}")
            
    except Exception as e:
        print(f"📡 Could not connect to Dashboard at {DASHBOARD}. Manual start required.")

# Run the initialization
def main():
    initialize_mission()

    # =========================================================
    # BANDWIDTH & LATENCY TRACKING (resets on every restart)
    # =========================================================
    total_raw_bytes = 0       # Raw bytes that WOULD have been downlinked
    total_alert_bytes = 0     # Compact alert bytes actually "sent"
    total_observations = 0    # Total lake passes with valid imagery
    normal_count = 0          # Passes classified as normal (no downlink)
    anomaly_count = 0         # Passes classified as red/orange (alert sent)
    last_tier1_s = 0.0        # Last Tier 1 (spectral math) duration
    last_tier2_s = None       # Last Tier 2 (VLM) duration — None if not triggered
    last_pipeline_s = 0.0     # Last end-to-end pipeline duration

    # =========================================================
    # MAIN LOOP
    # =========================================================

    print("Water Quality Monitoring Satellite Intelligence System Starting...\n")

    while True:

        try:
            # -------------------------------------------------
            # GET SIMULATION STATE (Time & Actual Position)
            # -------------------------------------------------
            sim_data = get_position()
            timestamp = sim_data["timestamp"]
            coords = sim_data["lon-lat-alt"] 
            
            sat_lon = coords[0] # First item is longitude
            sat_lat = coords[1] # Second item is latitude
            sat_alt = coords[2] # Third item is altitude 

            is_land = globe.is_land(sat_lat, sat_lon)
            surface = "LAND" if is_land else "OCEAN"

            # COUNTRY DETECTION - If the satellite is over the ocean, it will return the nearest country.
            location_data = reverse_geocode.get((sat_lat, sat_lon))
            country_name = location_data.get('country', 'Unknown')
            city_name = location_data.get('city', 'Unknown')
            
            safe_time = safe_timestamp(timestamp)

            print(f"\nSatellite Time: {timestamp} | Position: {sat_lat:.2f}, {sat_lon:.2f}")
            print(f"Surface: {surface} | Nearest Territory: {city_name}, {country_name}")

            # Logic: Find nearest asset and determine status
            current_status = "SCANNING..."
            status_color = "gray"

            sat_state = {
                "lon-lat-alt": [sat_lon, sat_lat, sat_alt],
                "timestamp": timestamp,
                "surface": surface
            }
            with open("mission_control/satellite_state.json", "w") as f:
                json.dump(sat_state, f)
        
            # -------------------------------------------------
            # LOOP OVER ASSETS
            # -------------------------------------------------
            for asset in ASSETS:

                name = asset["name"]
                
                # REALISM CHECK: Wait & Trigger 
                distance = haversine_distance(sat_lat, sat_lon, asset["lat"], asset["lon"])
                
                if distance > TRIGGER_RADIUS_KM:
                    continue
                
                print(f"\nTARGET IN RANGE: {name} ({distance:.1f}km)")
                asset_dir = os.path.join(SAVE_DIR, name)
                os.makedirs(asset_dir, exist_ok=True)

                # =================================================
                # FETCH IMAGE
                # =================================================

                current_path = os.path.join(
                    asset_dir,
                    f"{safe_time}.png"
                )

                print(f"IN RANGE: {name} ({distance:.1f}km). Triggering Live On-Board Camera...")

                # Fetches the image 
                img_resp = requests.get(
                    f"{BASE_URL}/data/image/sentinel",
                    params={
                        "lat": asset["lat"], "lon": asset["lon"], 
                        "timestamp": timestamp,
                        "spectral_bands": ['green', 'red', 'nir', 'rededge1'],
                        "size_km": 20.0,
                        "return_type": "array",
                        "window_seconds": 2592000
                    }
                )
                print(f"img_res.status_code: {img_resp.status_code} | Image hash: {hash(img_resp.content)}")
                print("HEADERS:", img_resp.headers.get("Content-Type"))
                print("TEXT:", img_resp.text[:500])
                
                save_path = os.path.join(
                    asset_dir,
                    f"{safe_time}.json"
                )
                if img_resp.status_code == 200 and len(img_resp.content) > 100:
                    print(f"Live Frame Captured over {country_name}")
                    with open(save_path, "w") as f:
                        f.write(img_resp.text)
                    print(f"Saved: {save_path}")  
                    data = img_resp.json()
                    print("img_resp.content type: ", type(img_resp.content))
                    print("img_resp.json() type: ", type(img_resp.json()))

                    # -------------------------------------------------
                    # BANDWIDTH: Raw image size from metadata
                    # uint16 = 2 bytes per pixel, shape = [bands, H, W]
                    # -------------------------------------------------
                    shape = data["image"]["metadata"]["shape"]
                    raw_bytes = shape[0] * shape[1] * shape[2] * 2
                    total_raw_bytes += raw_bytes
                    total_observations += 1
                    
                    # -------------------------------------------------
                    # TIER 1: Spectral analysis (timed)
                    # -------------------------------------------------
                    t0 = time.time()
                    metrics, risk_label, risk_color, should_trigger_vlm, arr, bands = analyze_water_risk(data)
                    t1 = time.time()
                    last_tier1_s = round(t1 - t0, 4)
                    last_tier2_s = None
                    last_pipeline_s = last_tier1_s
                    print(f"Metrics: turbidity={metrics['turbidity']:.4f} algae={metrics['algae_index']:.4f} water={metrics['water_score']:.4f} nir={metrics['nir_absorb']:.4f}")

                    
                    # -------------------------------------------------
                    # TIER 2: VLM (conditional, timed)
                    # -------------------------------------------------
                    vlm_description = "VLM not required."
                    if should_trigger_vlm and arr is not None:
                        t2 = time.time()
                        vlm_description = call_liquid_vlm(metrics, risk_label, arr, bands)
                        t3 = time.time()
                        last_tier2_s = round(t3 - t2, 2)
                        last_pipeline_s = round(t3 - t0, 2)

                    # -------------------------------------------------
                    # BANDWIDTH: Classification counts
                    # -------------------------------------------------
                    if risk_color in ('red', 'orange'):
                        anomaly_count += 1
                    else:
                        normal_count += 1

                    # -------------------------------------------------
                    # BANDWIDTH: Alert payload size (what we actually "send")
                    # Only the compact alert fields, not the full state object
                    # -------------------------------------------------
                    alert_payload = {
                        "name": name,
                        "status": risk_label,
                        "color": risk_color,
                        "vlm": vlm_description,
                        "timestamp": timestamp
                    }
                    alert_bytes = len(json.dumps(alert_payload).encode('utf-8'))
                    if risk_color in ('red', 'orange'):
                        total_alert_bytes += alert_bytes

                    saved_pct = round(
                        (1 - total_alert_bytes / total_raw_bytes) * 100, 2
                    ) if total_raw_bytes > 0 else 0.0

                    print(
                        f"📊 Bandwidth | Raw: {raw_bytes/1e6:.1f} MB  "
                        f"Alert: {alert_bytes} B  "
                        f"Saved: {saved_pct}%  "
                        f"[{normal_count} normal / {anomaly_count} anomalies]"
                    )
                    print(
                        f"⏱  Latency  | Tier 1: {last_tier1_s*1000:.1f} ms  "
                        + (f"Tier 2: {last_tier2_s:.2f} s  " if last_tier2_s else "")
                        + f"Pipeline: {last_pipeline_s:.2f} s"
                    )

                    # Update State for Dashboard
                    current_status = f"TARGET: {name} \nSTATUS: {risk_label} \nVLM INSIGHTS: {vlm_description}"
                    status_color = risk_color
                    name = asset["name"]
                    asset_state[name].update({
                        "status": risk_label,
                        "color": risk_color,
                        "updated_at": timestamp
                    })
                    # Convert dictionary back into a list for the map traces
                    assets_list_for_frontend = [
                        {"name": name, **data} 
                        for name, data in asset_state.items()
                    ]

                    # Prepare the final JSON structure
                    mission_state = {
                        "lon-lat-alt": [sat_lon, sat_lat, sat_alt],
                        "timestamp": timestamp,
                        "status": current_status,
                        "color": status_color,
                        "assets": assets_list_for_frontend,
                        # -----------------------------------------
                        # BANDWIDTH & LATENCY STATS FOR DASHBOARD
                        # -----------------------------------------
                        "bandwidth": {
                            "total_raw_mb": round(total_raw_bytes / 1_000_000, 2),
                            "total_alert_kb": round(total_alert_bytes / 1_000, 2),
                            "saved_pct": saved_pct,
                            "observations": total_observations,
                            "normal": normal_count,
                            "anomalies": anomaly_count
                        },
                        "latency": {
                            "last_pipeline_s": last_pipeline_s,
                            "last_tier1_ms": round(last_tier1_s * 1000, 1),
                            "last_tier2_s": last_tier2_s  # None if VLM not triggered
                        }
                    }

                    with open("mission_control/shared_state.json", "w") as f:
                        json.dump(mission_state, f)
                    print(f"Name: {asset['name']} | Risk: {risk_label} | Color: {risk_color}")
                    print(f"Status: {current_status}")
                
                elif img_resp.status_code != 200:
                    print("❌ Failed to fetch image")
                    continue
                else:
                    print("❌ Failed to fetch useful image")
                    continue

            time.sleep(5) 

        except Exception as e:
            print("Error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()

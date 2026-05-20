import os
import time
import requests
import pandas as pd
from tqdm import tqdm
import concurrent.futures # Imported for multi-threading capabilities

# --- CONFIGURATION ---
USERNAME = "hoangnhu1406"
PASSWORD = "Nhu@1406"

CSV_FILE_PATH = "data_splits/mimic-cxr-sub-img-edema-split-manualtest.csv"
OUTPUT_DIR = "data/img_data/"
FAILED_LOG_FILE = "failed_downloads.txt"
BASE_URL = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
MAX_WORKERS = 8  # Number of concurrent downloads (8 lanes instead of 1)

def construct_url(subject_id, study_id, dicom_id):
    """Constructs the explicit direct download URL for PhysioNet."""
    subject_id_str = str(subject_id)
    study_id_str = str(study_id)
    
    # folder_p1: 'p' + first 2 digits of subject_id
    folder_p1 = f"p{subject_id_str[:2]}"
    
    # folder_p2: 'p' + full subject_id
    folder_p2 = f"p{subject_id_str}"
    
    # folder_s: 's' + full study_id
    folder_s = f"s{study_id_str}"
    
    # filename: dicom_id + '.jpg'
    filename = f"{dicom_id}.jpg"
    
    url = f"{BASE_URL}{folder_p1}/{folder_p2}/{folder_s}/{filename}"
    return url

def download_image_task(session, url, output_path, dicom_id):
    """Downloads an image and returns the result status for the thread pool."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, stream=True, timeout=15)
            response.raise_for_status()  # Check for HTTP errors
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Return the dicom_id and a True flag indicating success
            return (dicom_id, True)
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                pass # Already reached max retries, fail and log
            else:
                time.sleep(RETRY_DELAY)
    
    # Return the dicom_id and a False flag indicating failure
    return (dicom_id, False)

def authenticate_session(username, password):
    """Authenticates and returns a session object."""
    session = requests.Session()
    login_url = "https://physionet.org/login/"
    
    try:
        # First, get the login page to retrieve the CSRF token
        response = session.get(login_url)
        response.raise_for_status()
        
        # The CSRF token is usually stored in the cookies
        csrftoken = session.cookies.get('csrftoken')
        
        if not csrftoken:
             print("[Warning] Could not retrieve CSRF token. Login might fail.")

        # Prepare login data
        login_data = {
            'username': username,
            'password': password,
            'csrfmiddlewaretoken': csrftoken,
            'next': '/files/mimic-cxr-jpg/2.0.0/' # Where to go after login
        }
        
        # Headers are important to act like a real browser
        headers = {
            'Referer': login_url,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Send login POST request
        post_response = session.post(login_url, data=login_data, headers=headers)
        post_response.raise_for_status()
        
        # A successful login usually redirects or doesn't have the login form again
        if "Sign in to your account" in post_response.text:
             print("[Error] Login failed. Please check your username and password.")
             return None
        
        print("[Success] Successfully authenticated session.")
        return session
        
    except requests.exceptions.RequestException as e:
        print(f"[Error] Failed to authenticate: {e}")
        return None

def main():
    # 1. Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Check Input CSV
    if not os.path.exists(CSV_FILE_PATH):
        print(f"[Error] CSV file not found at {CSV_FILE_PATH}")
        return
        
    df = pd.read_csv(CSV_FILE_PATH)
    failed_dicoms = []
    
    # 3. Setup authentication properly
    print("Authenticating with PhysioNet...")
    session = authenticate_session(USERNAME, PASSWORD)
    
    if not session:
         print("[Error] Aborting download due to authentication failure.")
         return

    # 4. Filter out images that are already downloaded (Resume feature)
    print("Scanning for existing files to skip...")
    tasks = []
    for index, row in df.iterrows():
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        
        output_filename = f"{dicom_id}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # If file already exists and is not empty, skip adding it to the download queue
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            continue
            
        url = construct_url(subject_id, study_id, dicom_id)
        # Append the specific details needed for the task
        tasks.append((url, output_path, dicom_id))

    if not tasks:
        print("[Success] All images have already been downloaded!")
        return

    print(f"Starting multi-threaded download of {len(tasks)} missing images...")
    
    # 5. Process downloads concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map tasks to the executor
        futures = [
            executor.submit(download_image_task, session, task[0], task[1], task[2]) 
            for task in tasks
        ]
        
        # Use tqdm to track completed futures as they finish
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading Images (Multi-thread)"):
            dicom_id, success = future.result()
            
            # Track failures
            if not success:
                failed_dicoms.append(dicom_id)
            
    # 6. Log permanently failed dicom_ids
    if failed_dicoms:
        with open(FAILED_LOG_FILE, "w") as f:
            for d_id in failed_dicoms:
                f.write(f"{d_id}\n")
        print(f"\n[Finished] Completed with {len(failed_dicoms)} errors. See {FAILED_LOG_FILE} for details.")
    else:
        print("\n[Finished] Successfully downloaded all images!")

if __name__ == "__main__":
    main()
This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

## Additional Info

# Directory Structure
```
.gitignore
.python-version
hello.py
image_processing_agent.py
pyproject.toml
ruler_calibration_tool.py
test_vertexai.py
```

# Files

## File: .gitignore
```
# GCP Credentials
*-my-project-credentials.json
*.json

# Image files
*.jpg
*.jpeg
*.png
*.gif
*.bmp
*.tiff

# Python virtual environment
.venv/
__pycache__/
*.pyc

# IDE specific files
.vscode/
.idea/
*.swp
*.swo
```

## File: .python-version
```
3.12
```

## File: hello.py
```python
def main():
    print("Hello from agent!")


if __name__ == "__main__":
    main()
```

## File: image_processing_agent.py
```python
# image_processing_agent.py

import os
from dotenv import load_dotenv
import uuid
import json
from google.cloud import storage
import asyncio # For asynchronous operations later
import concurrent.futures # For running blocking I/O in a separate thread

# --- Load Environment Variables ---
load_dotenv()

# --- Vertex AI and Gemini Imports ---
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part, FinishReason # Corrected import path
    import vertexai.preview.generative_models as preview_generative_models # For specific safety settings if needed
except ImportError:
    print("Failed to import Vertex AI libraries. Ensure 'google-cloud-aiplatform' is installed.")
    vertexai = None
    GenerativeModel = None

# Import the refactored calibration tool
from ruler_calibration_tool import calibrate_ruler_from_image_data

# --- Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")

# --- Initialize Vertex AI ---
try:
    if vertexai and GCP_PROJECT_ID and GCP_REGION:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        print(f"Vertex AI initialized for project '{GCP_PROJECT_ID}' in region '{GCP_REGION}'.")
    else:
        if not vertexai:
            print("Vertex AI library not loaded.")
        if not GCP_PROJECT_ID:
            print("WARNING: GCP_PROJECT_ID not set in environment. Vertex AI cannot be initialized.")
        if not GCP_REGION:
            print("WARNING: GCP_REGION not set in environment. Vertex AI cannot be initialized.")
except Exception as e:
    print(f"Error initializing Vertex AI: {e}")
    vertexai = None # Ensure it's None if init fails

# --- End Configuration ---

class ImageProcessingAgent:
    def __init__(self):
        self.storage_client = None
        self.gemini_model = None
        self.gcs_upload_timeout_seconds = 180
        self.gcs_upload_log = [] # To store logs for async uploads
        # ThreadPoolExecutor for non-blocking GCS uploads
        self.thread_pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2) # Adjust workers as needed

        try:
            self.storage_client = storage.Client()
            print("Storage client initialized.")
        except Exception as e:
            print(f"Error initializing Storage client: {e}. Ensure GOOGLE_APPLICATION_CREDENTIALS is set.")

        if vertexai and GenerativeModel:
            try:
                # Use a stable Gemini model available on Vertex AI
                # 'gemini-1.0-pro' is a good general choice.
                # 'gemini-1.5-flash-001' or 'gemini-1.5-pro-001' are newer options
                self.gemini_model_name = "gemini-1.0-pro" # Or "gemini-1.5-flash-001"
                self.gemini_model = GenerativeModel(self.gemini_model_name)
                print(f"Vertex AI Gemini model '{self.gemini_model_name}' initialized.")
            except Exception as e:
                print(f"Error initializing Vertex AI Gemini model: {e}")
                self.gemini_model = None
        else:
            print("Vertex AI or GenerativeModel not available. Summarization will be skipped.")

        print("ImageProcessingAgent initialized.")
        if not GCS_BUCKET_NAME:
            print("WARNING: GCS_BUCKET_NAME is not set. Image uploads will fail.")
        if not self.storage_client:
            print("WARNING: Google Cloud Storage client failed to initialize.")


    def _upload_to_gcs_sync(self, image_bytes, original_filename="ruler_image"): # Renamed to indicate it's sync internally
        if not self.storage_client:
            msg = "Storage client not initialized. Cannot upload to GCS."
            self.gcs_upload_log.append(f"ERROR: {msg}")
            print(msg)
            return None
        if not GCS_BUCKET_NAME:
            msg = "GCS_BUCKET_NAME not configured. Cannot upload."
            self.gcs_upload_log.append(f"ERROR: {msg}")
            print(msg)
            return None

        try:
            bucket = self.storage_client.bucket(GCS_BUCKET_NAME)
            image_id = str(uuid.uuid4())
            content_type = 'image/jpeg'
            if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                content_type = 'image/png'
            elif image_bytes.startswith(b'\xff\xd8\xff'): # Basic JPEG check
                pass # Default is jpeg

            file_extension = content_type.split('/')[1]
            filename = f"{original_filename}_{image_id}.{file_extension}"
            
            blob = bucket.blob(filename)
            
            upload_timeout_seconds = 180
            log_msg_start = f"Attempting to upload {len(image_bytes)/(1024*1024):.2f} MB as {filename} with timeout {self.gcs_upload_timeout_seconds}s..." # Immediate feedback
            # self.gcs_upload_log.append(log_msg_start) # Log start attempt

            blob.upload_from_string(
                image_bytes,
                content_type=content_type,
                timeout=self.gcs_upload_timeout_seconds
            )
            gcs_uri = f"gs://{GCS_BUCKET_NAME}/{filename}"
            success_msg = f"Image successfully uploaded to {gcs_uri}"
            self.gcs_upload_log.append(f"SUCCESS: {success_msg}")
            print(success_msg)
            return gcs_uri
        except Exception as e:
            error_msg = f"Error uploading to GCS: {e}"
            self.gcs_upload_log.append(f"ERROR: {error_msg}")
            print(error_msg)
            return None

    async def _upload_to_gcs_async(self, image_bytes, original_filename="ruler_image"):
        """Wraps the synchronous GCS upload in an async-compatible way using run_in_executor."""
        loop = asyncio.get_running_loop()
        # Use partial to pass arguments to the sync function
        # functools.partial can also be used here
        upload_task = loop.run_in_executor(
            self.thread_pool_executor, 
            self._upload_to_gcs_sync, # Call the synchronous version
            image_bytes, 
            original_filename
        )
        # We don't await it here; we let it run in the background.
        # The calling function will manage the future.
        return upload_task


    def _get_calibration_results(self, image_bytes, ruler_length_cm=15.0, targets=["0", "15"]):
        print(f"Requesting calibration with length {ruler_length_cm}cm and targets {targets}")
        results = calibrate_ruler_from_image_data(image_bytes, ruler_length_cm, targets)
        return results

    def _summarize_with_vertex_gemini(self, calibration_data, ruler_actual_length_cm):
        if not self.gemini_model:
            print("Vertex AI Gemini model not available. Skipping summarization.")
            return "Summarization skipped (Vertex AI Gemini model not available)."

        if not calibration_data or calibration_data.get("status") != "success":
            return "Could not generate summary due to calibration failure or no calibration data."

        prompt_parts = [
            "You are an analytical assistant. Based on the following image calibration data for a ruler, provide a concise summary.",
            "The goal was to determine the image scale (pixels per cm and pixels per inch).",
            f"Calibration Status: {calibration_data.get('message')}",
            f"Target Markers Used: {list(calibration_data.get('found_target_coordinates', {}).keys())}",
            f"Coordinates Found: {json.dumps(calibration_data.get('found_target_coordinates'), indent=2)}",
            f"Pixel Distance Between Markers: {calibration_data.get('pixel_distance', 'N/A')} pixels",
            f"Assumed Ruler Length for these markers: {ruler_actual_length_cm} cm", # Passed in
            f"Calculated Pixels per CM: {calibration_data.get('pixels_per_cm', 'N/A')}",
            f"Calculated Pixels per Inch: {calibration_data.get('pixels_per_inch', 'N/A')}",
            "\nSummarize the key findings regarding the image scale calibration. Was it successful? What is the determined scale?",
            "If there were issues, briefly mention them."
        ]
        prompt = "\n".join(prompt_parts)

        print(f"\nSending prompt to Vertex AI Gemini ({self.gemini_model_name}):\n{prompt[:500]}...\n")

        try:
            # Define safety settings if needed, otherwise defaults are used
            safety_settings = {
                preview_generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: preview_generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                preview_generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: preview_generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                preview_generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: preview_generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                preview_generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: preview_generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            response = self.gemini_model.generate_content(
                prompt,
                # generation_config={"temperature": 0.7}, # Example config
                # safety_settings=safety_settings # Optional
            )
            
            if response.candidates and response.candidates[0].content.parts:
                summary = response.candidates[0].content.parts[0].text
            else:
                summary = "Gemini response received, but contained no processable parts or was blocked."
                if response.prompt_feedback:
                     summary += f" Prompt Feedback: {response.prompt_feedback.block_reason}"
                     if response.prompt_feedback.block_reason_message:
                        summary += f" ({response.prompt_feedback.block_reason_message})"
                for candidate in response.candidates:
                    if candidate.finish_reason != FinishReason.STOP:
                        summary += f" Candidate Finish Reason: {candidate.finish_reason.name}"
                        if candidate.finish_message:
                             summary += f" ({candidate.finish_message})"


            print(f"Vertex AI Gemini Summary Received.")
            return summary
        except Exception as e:
            print(f"Error calling Vertex AI Gemini API: {e}")
            return f"Error generating summary with Vertex AI Gemini: {str(e)}"

    async def process_image_and_summarize_async(self, image_bytes, ruler_actual_length_cm=15.0, target_texts_on_ruler=["0", "15"]):
        if not self.storage_client or not vertexai: # Check Vertex AI init too
            error_msg = "Agent not properly initialized (storage client or Vertex AI missing). Cannot process."
            self.gcs_upload_log.append(f"ABORT: {error_msg}")
            return {"error": error_msg, "gcs_upload_log": self.gcs_upload_log}
            
        print("\n--- Agent: Starting Image Processing (Async GCS Upload) ---")
        
        # 1. Start GCS upload asynchronously
        print("Initiating asynchronous image upload to GCS...")
        gcs_upload_future = await self._upload_to_gcs_async(image_bytes, "ruler_image")
        # The upload now runs in the background via the thread pool.

        # 2. Perform calibration (this is CPU/Vision API bound, can run while GCS uploads)
        print("Performing ruler calibration...")
        calibration_results = self._get_calibration_results(image_bytes, ruler_actual_length_cm, target_texts_on_ruler)
        print(f"Calibration Tool Results: {json.dumps(calibration_results, indent=2)}")

        summary = "Summary generation skipped due to calibration failure or other issues."
        if calibration_results.get("status") == "success":
            # 3. Summarize the response with Gemini (after calibration is done)
            print("Summarizing calibration results with Vertex AI Gemini...")
            summary = self._summarize_with_vertex_gemini(calibration_results, ruler_actual_length_cm) # Pass ruler_length
        else:
            self.gcs_upload_log.append(f"INFO: Calibration failed, summary not generated. Details: {calibration_results.get('message')}")
            print(f"Calibration failed: {calibration_results.get('message')}")


        # 4. Wait for GCS upload to complete (if not already) and get its result
        gcs_path = None
        print("Checking GCS upload status...")
        try:
            # Wait for the upload task to complete.
            # The timeout here is for how long this main thread waits for the already running background thread.
            # The background thread itself has its own GCS client timeout.
            gcs_path = await asyncio.wait_for(gcs_upload_future, timeout=self.gcs_upload_timeout_seconds + 10) # Slightly more than GCS internal timeout
            if gcs_path:
                 print(f"Async GCS upload confirmed complete. Path: {gcs_path}")
            else:
                 print("Async GCS upload completed but returned no path (likely an error logged by the upload function).")
        except asyncio.TimeoutError:
            error_msg = "Timeout waiting for GCS upload thread to complete. Upload might still be in progress or failed."
            self.gcs_upload_log.append(f"TIMEOUT_WAIT: {error_msg}")
            print(error_msg)
        except Exception as e:
            error_msg = f"Exception while waiting for GCS upload result: {e}"
            self.gcs_upload_log.append(f"EXCEPTION_WAIT: {error_msg}")
            print(error_msg)
        
        print("--- Agent: Image Processing Complete ---")
        final_result = {
            "status": "success" if calibration_results.get("status") == "success" else "processing_error",
            "gcs_path": gcs_path, # This will be None if upload failed or timed out here
            "calibration_data": calibration_results,
            "summary": summary,
            "gcs_upload_log": self.gcs_upload_log # Include the log
        }
        if calibration_results.get("status") != "success":
            final_result["error_details"] = calibration_results.get("message")
        return final_result

    def shutdown(self):
        print("Shutting down thread pool executor...")
        self.thread_pool_executor.shutdown(wait=True)
        print("Thread pool executor shut down.")


# --- Main Execution (for testing this agent directly) ---
async def main_async_test(): # Make the main test function async
    print("Starting ASYNCHRONOUS image processing agent test run...")

    if not GCP_PROJECT_ID or not GCP_REGION:
        print("ERROR: GCP_PROJECT_ID or GCP_REGION not configured. Vertex AI cannot run.")
        return
    if not GCS_BUCKET_NAME:
        print("ERROR: GCS_BUCKET_NAME is not configured.")
        return
    
    test_image_path = 'ruler_image.jpg'
    test_ruler_len = 15.0
    test_ruler_targets = ["0", "15"]

    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image '{test_image_path}' not found.")
        return

    try:
        with open(test_image_path, 'rb') as f:
            test_image_bytes = f.read()
        print(f"Loaded test image '{test_image_path}' ({len(test_image_bytes)} bytes)")

        agent = ImageProcessingAgent()
        if agent.storage_client and agent.gemini_model :
            result = await agent.process_image_and_summarize_async( # await the async method
                test_image_bytes,
                ruler_actual_length_cm=test_ruler_len,
                target_texts_on_ruler=test_ruler_targets
            )
            
            print("\n--- Final Agent Output ---")
            print(json.dumps(result, indent=2))
        else:
            print("Agent could not be initialized properly (check GCS/Vertex AI setup and credentials). Test aborted.")
        
        agent.shutdown() # Important to clean up the thread pool

    except Exception as e:
        print(f"An error occurred during the agent test run: {e}")
        # If agent was created, try to shut down its executor
        if 'agent' in locals() and hasattr(agent, 'shutdown'):
            agent.shutdown()

if __name__ == "__main__":
    # To run the async main function
    asyncio.run(main_async_test())
```

## File: pyproject.toml
```toml
[project]
name = "agent"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "google-adk>=0.5.0",
    "google-cloud-aiplatform>=1.92.0",
    "google-cloud-storage>=2.19.0",
    "google-cloud-vision>=3.10.1",
    "google-generativeai>=0.8.5",
    "python-dotenv>=1.1.0",
]
```

## File: ruler_calibration_tool.py
```python
import io
import os # Keep os for potential future use, but not strictly needed for core logic now
from google.cloud import vision
import math

# It's good practice to initialize the client once,
# potentially outside the main function if this module is imported.
# If the GOOGLE_APPLICATION_CREDENTIALS env var is set, this will work.
try:
    client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Error initializing Vision API client: {e}")
    print("Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly.")
    client = None # Set to None so functions can check

def get_text_coordinates_from_bytes(image_content_bytes, target_texts):
    """
    Detects text in an image (provided as bytes) and returns the coordinates of target texts.
    Returns a dictionary like: {'text': (center_x, center_y), ...} or None on error.
    """
    if not client:
        print("Vision API client not initialized.")
        return None, "Vision API client not initialized."

    if not image_content_bytes:
        return None, "No image content provided."

    found_coordinates = {}
    detected_texts_info = [] # To store info about all detected texts for debugging

    try:
        image = vision.Image(content=image_content_bytes)
        response = client.text_detection(image=image) # Use the global client

        if response.error.message:
            error_message = (
                f'Vision API Error: {response.error.message}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'
            )
            print(error_message)
            return None, error_message

        texts = response.text_annotations

        if texts:
            # print(f"Found {len(texts)} text elements (raw):") # Debug
            for text_annotation in texts[1:]:  # Skip the first one (full text block)
                desc = text_annotation.description.strip().lower()
                vertices_repr = [(v.x, v.y) for v in text_annotation.bounding_poly.vertices]
                detected_texts_info.append({"description": desc, "vertices": vertices_repr})
                # print(f"- Description: '{desc}', Vertices: {vertices_repr}") # Debug

                for target in target_texts:
                    target_lower = target.lower()
                    # Try exact match or if target is part of the detected text (e.g., "0cm" contains "0")
                    if target_lower == desc or target_lower in desc:
                        x_coords = [vertex.x for vertex in text_annotation.bounding_poly.vertices]
                        y_coords = [vertex.y for vertex in text_annotation.bounding_poly.vertices]
                        center_x = sum(x_coords) / len(x_coords)
                        center_y = sum(y_coords) / len(y_coords)

                        if target not in found_coordinates: # Store the first match found
                            found_coordinates[target] = (center_x, center_y)
                            # print(f"  -> Matched '{target}' at center: ({center_x:.2f}, {center_y:.2f})") # Debug
                        if len(found_coordinates) == len(target_texts): # Optimization
                            break
                if len(found_coordinates) == len(target_texts): # Optimization
                    break
        else:
            print("No text detected in the image by Vision API.")
            return {}, "No text detected in the image." # Return empty dict for coords, specific message

        return found_coordinates, "Text detection complete."

    except Exception as e:
        error_msg = f"Error during text detection: {e}"
        print(error_msg)
        return None, error_msg

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x1, y1) and (x2, y2)."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calibrate_ruler_from_image_data(image_content_bytes, ruler_actual_length_cm, target_texts_on_ruler):
    """
    Main function to perform ruler calibration from image byte data.

    Args:
        image_content_bytes (bytes): The byte content of the image.
        ruler_actual_length_cm (float): The actual length of the ruler being measured (e.g., 15.0 for a 15cm ruler).
        target_texts_on_ruler (list): A list of two strings representing the start and end markings
                                      on the ruler to find (e.g., ["0", "15"] or ["0cm", "15cm"]).

    Returns:
        dict: A dictionary containing calibration results:
              {
                  "status": "success" or "failure",
                  "message": "Descriptive message",
                  "detected_texts_info": [{"description": "text", "vertices": [(x,y), ...]}, ...], (Optional, for debugging)
                  "found_target_coordinates": {"text_marker": (x, y), ...},
                  "pixel_distance": float or None,
                  "pixels_per_cm": float or None,
                  "pixels_per_inch": float or None
              }
    """
    if not client:
        return {
            "status": "failure",
            "message": "Vision API client not initialized. Cannot proceed.",
            "detected_texts_info": [],
            "found_target_coordinates": {},
            "pixel_distance": None,
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

    if not image_content_bytes:
        return {
            "status": "failure",
            "message": "No image data provided for calibration.",
            "detected_texts_info": [],
            "found_target_coordinates": {},
            "pixel_distance": None,
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

    if not (isinstance(target_texts_on_ruler, list) and len(target_texts_on_ruler) == 2):
        return {
            "status": "failure",
            "message": "target_texts_on_ruler must be a list of two strings (start and end markers).",
            "detected_texts_info": [],
            "found_target_coordinates": {},
            "pixel_distance": None,
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

    print(f"Attempting to find coordinates for: {target_texts_on_ruler}")
    coordinates, text_detection_message = get_text_coordinates_from_bytes(image_content_bytes, target_texts_on_ruler)

    if coordinates is None: # Indicates a significant error in get_text_coordinates_from_bytes
        return {
            "status": "failure",
            "message": f"Could not get text coordinates. Detail: {text_detection_message}",
            "detected_texts_info": [], # Could populate with partial info if available
            "found_target_coordinates": {},
            "pixel_distance": None,
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

    # Check if both target texts were found
    start_marker, end_marker = target_texts_on_ruler[0], target_texts_on_ruler[1]
    if start_marker not in coordinates or end_marker not in coordinates:
        message = (f"Could not find one or both target texts ('{start_marker}', '{end_marker}') with sufficient clarity. "
                   f"Found: {list(coordinates.keys())}. Text detection status: {text_detection_message}")
        # print(message) # Also print to console for immediate feedback during dev
        return {
            "status": "failure",
            "message": message,
            "detected_texts_info": [], # You might want to return detected_texts_info from get_text_coordinates_from_bytes here
            "found_target_coordinates": coordinates,
            "pixel_distance": None,
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

    point_start = coordinates[start_marker]
    point_end = coordinates[end_marker]

    pixel_dist = calculate_distance(point_start, point_end)

    if pixel_dist > 0:
        px_per_cm = pixel_dist / ruler_actual_length_cm
        px_per_inch = px_per_cm * 2.54  # 1 inch = 2.54 cm
        return {
            "status": "success",
            "message": "Calibration successful.",
            # "detected_texts_info": detected_texts_info, # Optionally include all detected texts
            "found_target_coordinates": coordinates,
            "pixel_distance": round(pixel_dist, 2),
            "pixels_per_cm": round(px_per_cm, 2),
            "pixels_per_inch": round(px_per_inch, 2)
        }
    else:
        return {
            "status": "failure",
            "message": "Pixel distance between markers is zero. Cannot calculate scale.",
            # "detected_texts_info": detected_texts_info,
            "found_target_coordinates": coordinates,
            "pixel_distance": round(pixel_dist, 2),
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

# --- Example of how to use this module (for testing) ---
if __name__ == "__main__":
    print("Testing ruler_calibration_tool.py...")

    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set in your environment
    # For this test, you need an image file named 'ruler_image.jpg' in the same directory,
    # or provide a different path.
    test_image_path = 'ruler_image.jpg' # The same image your original script used
    test_ruler_length = 15.0  # cm
    test_targets = ["0", "15"] # Modify based on your ruler's markings

    if not os.path.exists(test_image_path):
        print(f"Test image '{test_image_path}' not found. Skipping functional test.")
        # You could create dummy bytes for a more basic test:
        # dummy_image_bytes = b"fake image data"
        # results = calibrate_ruler_from_image_data(dummy_image_bytes, test_ruler_length, test_targets)
        # print("\nResults (with dummy data):")
        # import json
        # print(json.dumps(results, indent=2))
    else:
        try:
            with io.open(test_image_path, 'rb') as image_file:
                test_image_content_bytes = image_file.read()
            print(f"Loaded test image '{test_image_path}' ({len(test_image_content_bytes)} bytes)")

            results = calibrate_ruler_from_image_data(test_image_content_bytes, test_ruler_length, test_targets)

            print("\n--- Calibration Results ---")
            import json
            print(json.dumps(results, indent=2))

            if results["status"] == "success":
                print(f"\nSuccessfully calibrated: {results['pixels_per_cm']} pixels/cm")
            else:
                print(f"\nCalibration failed: {results['message']}")

        except Exception as e:
            print(f"Error during test execution: {e}")

    # Test case: No image data
    print("\n--- Testing with no image data ---")
    no_image_results = calibrate_ruler_from_image_data(None, 15.0, ["0", "15"])
    print(json.dumps(no_image_results, indent=2))

    # Test case: Invalid targets
    print("\n--- Testing with invalid target texts ---")
    invalid_targets_results = calibrate_ruler_from_image_data(b"dummy", 15.0, ["0"]) # Not a list of two
    print(json.dumps(invalid_targets_results, indent=2))
```

## File: test_vertexai.py
```python
import os
import vertexai
from vertexai.generative_models import GenerativeModel
from dotenv import load_dotenv

# --- Load Environment Variables (if you use a .env file) ---
load_dotenv()

# --- Configuration ---
# Replace with your actual project ID and region if not using environment variables
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "my-agentic-cv-project-459312")
GCP_REGION = os.getenv("GCP_REGION", "europe-west1") # Make sure this is where your model is available

def test_gemini():
    print(f"Attempting to initialize Vertex AI for project '{GCP_PROJECT_ID}' in region '{GCP_REGION}'...")
    try:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        print("Vertex AI initialized successfully.")
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        return

    try:
        # You can try 'gemini-1.0-pro-vision' if you intend to send images directly later,
        # but for a simple text prompt, 'gemini-1.0-pro' is fine.
        # Or try one of the newer models like 'gemini-1.5-flash-001'
        model_name = "gemini-1.5-flash-001" # or "gemini-1.5-flash-001"
        print(f"Loading GenerativeModel: {model_name}...")
        model = GenerativeModel(model_name)
        print("Model loaded successfully.")

        prompt = "What is the capital of France?"
        print(f"Sending prompt to Gemini: '{prompt}'")
        response = model.generate_content(prompt)

        print("Response from Gemini:")
        if response.candidates and response.candidates[0].content.parts:
            print(response.candidates[0].content.parts[0].text)
        else:
            print("No content in response or response was blocked.")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 print(f"Prompt Feedback: {response.prompt_feedback}")
            for candidate in response.candidates:
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                     print(f"Candidate Finish Reason: {candidate.finish_reason}")
                if hasattr(candidate, 'finish_message') and candidate.finish_message:
                     print(f"Candidate Finish Message: {candidate.finish_message}")


    except Exception as e:
        print(f"An error occurred while trying to use the Gemini model: {e}")

if __name__ == "__main__":
    if not GCP_PROJECT_ID:
        print("ERROR: GCP_PROJECT_ID not set. Please set it in your environment or in the script.")
    elif not GCP_REGION:
        print("ERROR: GCP_REGION not set. Please set it in your environment or in the script.")
    else:
        test_gemini()
```

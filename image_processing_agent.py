# image_processing_agent.py

import os
from dotenv import load_dotenv
import uuid
import json
from google.cloud import storage
import asyncio # For asynchronous operations later
import concurrent.futures # For running blocking I/O in a separate thread


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import openai 


# Import the refactored calibration tool
from ruler_calibration_tool import calibrate_ruler_from_image_data

# --- Configuration ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")


class ImageProcessingAgent:
    def __init__(self):
        self.storage_client = None
        self.openai_client = None
        self.openai_model_name = "gpt-4o-mini"
        self.gcs_upload_timeout_seconds = 180
        self.gcs_upload_log = [] # To store logs for async uploads
        # ThreadPoolExecutor for non-blocking GCS uploads
        self.thread_pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2) # Adjust workers as needed

        # Initialize OpenAI Client
        if OPENAI_API_KEY:
            try:
                self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
                print(f"OpenAI client initialized with model '{self.openai_model_name}'.")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
        else:
            print("WARNING: OPENAI_API_KEY not set. OpenAI summarization will not be available.")

        # Initialize GCS client
        try:
            self.storage_client = storage.Client()
            print("Storage client initialized.")
        except Exception as e:
            print(f"Error initializing Storage client: {e}. Ensure GOOGLE_APPLICATION_CREDENTIALS is set.")


        print("ImageProcessingAgent initialized.")
        if not GCS_BUCKET_NAME:
            print("WARNING: GCS_BUCKET_NAME is not set. Image uploads will fail.")
        if not self.storage_client:
            print("WARNING: Google Cloud Storage client failed to initialize.")
        if not self.openai_client:
            print("WARNING: OpenAI client failed to initialize.")


    def _upload_to_gcs_sync(self, image_bytes, original_filename="ruler_image"):
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

            log_msg_start = f"Attempting to upload {len(image_bytes)/(1024*1024):.2f} MB as {filename} with timeout {self.gcs_upload_timeout_seconds}s..."
            print(log_msg_start) # Immediate feedback for sync part too

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
        upload_task = loop.run_in_executor(
            self.thread_pool_executor,
            self._upload_to_gcs_sync,
            image_bytes,
            original_filename
        )
        return upload_task


    def _get_calibration_results(self, image_bytes, ruler_length_cm=15.0, targets=["0", "15"]):
        print(f"Requesting calibration with length {ruler_length_cm}cm and targets {targets}")
        results = calibrate_ruler_from_image_data(image_bytes, ruler_length_cm, targets)
        return results

    def _summarize_with_openai(self, calibration_data, ruler_actual_length_cm):
        if not self.openai_client:
            return "Summarization skipped (OpenAI client not available)."

        if not calibration_data or calibration_data.get("status") != "success":
            return "Could not generate summary due to calibration failure or no calibration data."

        prompt_messages = [
            {
                "role": "system",
                "content": "You are an analytical assistant. Based on the following image calibration data for a ruler, provide a concise summary. The goal was to determine the image scale (pixels per cm and pixels per inch)."
            },
            {
                "role": "user",
                "content": f"""
                Calibration Status: {calibration_data.get('message')}
                Target Markers Used: {list(calibration_data.get('found_target_coordinates', {}).keys())}
                Coordinates Found: {json.dumps(calibration_data.get('found_target_coordinates'), indent=2)}
                Pixel Distance Between Markers: {calibration_data.get('pixel_distance', 'N/A')} pixels
                Assumed Ruler Length for these markers: {ruler_actual_length_cm} cm
                Calculated Pixels per CM: {calibration_data.get('pixels_per_cm', 'N/A')}
                Calculated Pixels per Inch: {calibration_data.get('pixels_per_inch', 'N/A')}

                Summarize the key findings regarding the image scale calibration. Was it successful? What is the determined scale?
                If there were issues, briefly mention them.
                """
            }
        ]

        print(f"\nSending prompt to OpenAI ({self.openai_model_name})...\n")

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model_name,
                messages=prompt_messages,
                temperature=0.7, # Adjust as needed
                # max_tokens=150 # Adjust as needed
            )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                summary = response.choices[0].message.content.strip()
            else:
                summary = "OpenAI response received, but contained no processable content."
                # You might want to log more details from the response object if needed

            print("OpenAI Summary Received.")
            return summary
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"Error generating summary with OpenAI: {str(e)}"


    async def process_image_and_summarize_async(self, image_bytes, ruler_actual_length_cm=15.0, target_texts_on_ruler=["0", "15"]):
        # Initial check for critical components for processing (storage client is key, summarizer is secondary for this method)
        if not self.storage_client:
            error_msg = "Agent not properly initialized (storage client missing). Cannot process."
            self.gcs_upload_log.append(f"ABORT: {error_msg}")
            
            return {"error": error_msg, "gcs_upload_log": self.gcs_upload_log}

        print("\n--- Agent: Starting Image Processing (Async GCS Upload) ---")

        # 1. Start GCS upload asynchronously
        print("Initiating asynchronous image upload to GCS...")
        gcs_upload_future = await self._upload_to_gcs_async(image_bytes, "ruler_image")

        # 2. Perform calibration
        print("Performing ruler calibration...")
        calibration_results = self._get_calibration_results(image_bytes, ruler_actual_length_cm, target_texts_on_ruler)
        print(f"Calibration Tool Results: {json.dumps(calibration_results, indent=2)}")

        summary = "Summary generation skipped." # Default message
        if calibration_results.get("status") == "success":
            print("Summarizing calibration results...")
            if self.openai_client: # Prioritize OpenAI
                summary = self._summarize_with_openai(calibration_results, ruler_actual_length_cm)

            else:
                summary = "No summarization model (OpenAI or Gemini) available."
                print("WARNING: "+summary)
        else:
            summary_fail_reason = calibration_results.get("message", "Calibration failed or no data.")
            summary = f"Summary generation skipped due to: {summary_fail_reason}"
            self.gcs_upload_log.append(f"INFO: Calibration failed, summary not generated. Details: {summary_fail_reason}")
            print(f"Calibration failed: {summary_fail_reason}")


        # 4. Wait for GCS upload to complete and get its result
        gcs_path = None
        print("Checking GCS upload status...")
        try:
            gcs_path = await asyncio.wait_for(gcs_upload_future, timeout=self.gcs_upload_timeout_seconds + 10)
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
            "gcs_path": gcs_path,
            "calibration_data": calibration_results,
            "summary": summary,
            "gcs_upload_log": self.gcs_upload_log
        }
        if calibration_results.get("status") != "success":
            final_result["error_details"] = calibration_results.get("message")
        return final_result

    def shutdown(self):
        print("Shutting down thread pool executor...")
        self.thread_pool_executor.shutdown(wait=True)
        print("Thread pool executor shut down.")


# --- Main Execution (for testing this agent directly) ---
async def main_async_test():
    print("Starting ASYNCHRONOUS image processing agent test run...")

    # Essential config check for cloud services
    gcs_ready = bool(GCS_BUCKET_NAME)
    openai_ready = bool(OPENAI_API_KEY)

    if not gcs_ready:
        print("ERROR: GCS_BUCKET_NAME is not configured. GCS operations will fail.")
        # Depending on strictness, you might return here.

    if not openai_ready: # If neither summarizer can possibly be configured
        print("ERROR: OPENAI_API_KEY is not configured. Summarization will likely fail.")
        # return # Could exit if no summarizer can be initialized

    test_image_path = 'ruler_image.jpg' # Ensure this image exists for testing
    test_ruler_len = 15.0
    test_ruler_targets = ["0", "15"]

    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image '{test_image_path}' not found. Please create it or update the path.")
        # Create a dummy image for basic testing if needed
        # with open(test_image_path, 'wb') as f:
        #     f.write(b"dummy jpeg data") # Not a real image, Vision API will fail
        # print(f"INFO: Created a dummy '{test_image_path}' for testing structure. Vision API will fail with this.")
        return


    agent = ImageProcessingAgent() # Agent initialization prints its own warnings based on env vars

    try:
        with open(test_image_path, 'rb') as f:
            test_image_bytes = f.read()
        print(f"Loaded test image '{test_image_path}' ({len(test_image_bytes)} bytes)")

        # Check if agent is minimally viable for the test (has storage and at least one summarizer)
        if agent.storage_client and agent.openai_client:
            print("Agent appears to have necessary components (Storage and at least one LLM client).")
            result = await agent.process_image_and_summarize_async(
                test_image_bytes,
                ruler_actual_length_cm=test_ruler_len,
                target_texts_on_ruler=test_ruler_targets
            )

            print("\n--- Final Agent Output ---")
            print(json.dumps(result, indent=2))
        else:
            print("Agent could not be initialized with minimum required components (Storage client and/or an LLM client). Test aborted.")
            print(f"Agent state: storage_client_exists={bool(agent.storage_client)}, "
                  f"openai_client_exists={bool(agent.openai_client)}")

    except Exception as e:
        print(f"An error occurred during the agent test run: {e}")
    finally:
        # Ensure agent shutdown happens even if errors occur after agent creation
        if 'agent' in locals() and hasattr(agent, 'shutdown'):
            agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main_async_test())
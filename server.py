import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import zstandard
from datetime import datetime
import sys  # <-- Import sys to allow exiting the program

# Create a FastAPI application instance.
app = FastAPI(title="Artemis Data Snapshot Tool")

# Create a Zstd decompressor instance.
dctx = zstandard.ZstdDecompressor()


# --- Helper function to format bytes (unchanged) ---
def format_bytes(size_in_bytes: int) -> str:
    """Converts a byte count into a human-readable string (KB, MB, etc.)."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} KB"
    else:
        return f"{size_in_bytes / 1024**2:.2f} MB"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    This endpoint now accepts a single WebSocket message, saves the decompressed
    JSON payload to a file, and then gracefully exits the server.
    """
    await websocket.accept()
    print("[Server] Client connected. Waiting for a single context payload...")

    try:
        # --- MODIFIED: We no longer loop. We only process one message. ---

        # 1. Receive the raw bytes from the client.
        compressed_data = await websocket.receive_bytes()

        compressed_size = len(compressed_data)
        print("\n" + "=" * 20 + " CONTEXT PAYLOAD RECEIVED " + "=" * 20)
        print(
            f"[Info] Received compressed payload size: {format_bytes(compressed_size)}"
        )

        try:
            # 2. Decompress the data.
            with dctx.stream_reader(compressed_data) as reader:
                decompressed_data = reader.read()

            uncompressed_size = len(decompressed_data)
            print(f"[Info] Decompressed size: {format_bytes(uncompressed_size)}")

            # 3. Parse the JSON to ensure it's valid.
            parsed_json = json.loads(decompressed_data)

            # --- NEW: Save the data to a file ---
            # Create a unique filename with a timestamp.
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"context_snapshot_{timestamp}.json"

            # Write the pretty-printed JSON to the file.
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(parsed_json, f, indent=2, ensure_ascii=False)

            print(f"[SUCCESS] Context snapshot saved to '{output_filename}'")

        except zstandard.ZstdError as e:
            print(f"[ERROR] Zstd decompression failed: {e}")
        except json.JSONDecodeError:
            print("[ERROR] Received data is not valid JSON after decompression.")

        print("=" * 54 + "\n")

    except WebSocketDisconnect:
        print("[Server] Client disconnected before sending data.")
    except Exception as e:
        print(f"[Server] An unexpected error occurred: {e}")
    finally:
        # --- NEW: Gracefully close the connection and exit the application ---
        print("[Server] Task complete. Shutting down.")
        await websocket.close()
        # This is a forceful way to stop uvicorn. For a simple script, it's effective.
        sys.exit(0)


if __name__ == "__main__":
    print("Starting Artemis snapshot server on ws://0.0.0.0:8000/ws")
    uvicorn.run(app, host="0.0.0.0", port=8000)

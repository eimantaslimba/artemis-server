import uvicorn
import json
import zstandard
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from llama_cpp import Llama
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

# ==========================================
# 1. TYPE DEFINITIONS (Synced with Rust)
# ==========================================


class SizingInfo(BaseModel):
    x: int
    y: int
    width: int
    height: int


class UiElement(BaseModel):
    id: str
    control_type: str
    name: str
    automation_id: str
    sizing_info: SizingInfo
    flags: List[str] = []
    children: List[str] = []


class UiTree(BaseModel):
    hwnd: int
    root_id: str
    elements: Dict[str, UiElement]


class WindowInfo(BaseModel):
    hwnd: int
    title: str
    class_name: str
    sizing_info: SizingInfo
    pid: int
    tags: List[str] = []
    cdp_port: Optional[int] = None


class WindowSnapshot(BaseModel):
    info: WindowInfo
    contents: Optional[UiTree] = None


class SystemContext(BaseModel):
    windows: List[WindowSnapshot] = []


# ==========================================
# 2. GLOBAL STATE & MODEL
# ==========================================

# Holds the latest state of the user's PC
CURRENT_CONTEXT = SystemContext()

# Compression Context
dctx = zstandard.ZstdDecompressor()

# Load Gemma 3 (Path set by start script)
MODEL_PATH = "/workspace/model.gguf"

print(f"--- INITIALIZING GEMMA 3 27B ---")
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,  # Push ALL layers to GPU
        n_ctx=8192 * 4,  # Context
        verbose=False,
    )
    print("--- MODEL LOADED ---")
except Exception as e:
    print(f"!!! MODEL LOAD FAILED: {e}")
    llm = None

app = FastAPI()

# ==========================================
# 3. HELPER: Format Context for AI
# ==========================================


def build_prompt(user_query: str, context: SystemContext) -> str:
    """
    Intelligently flattens the UI Tree into a prompt Gemma can read.
    """
    system_desc = "Current System UI State:\n"

    if not context.windows:
        system_desc += "No active windows detected.\n"

    for i, win in enumerate(context.windows):
        info = win.info
        system_desc += f"Window {i}: '{info.title}' (App: {info.class_name})\n"

        # If we have UI contents, list the important bits
        if win.contents:
            system_desc += "  Key UI Elements:\n"
            count = 0
            # Iterate through elements, prioritize buttons/inputs
            for el_id, el in win.contents.elements.items():
                if count > 30:  # Hard limit to save context tokens
                    system_desc += "  ... (more elements)\n"
                    break

                # Filter: Only show interesting elements (Buttons, Edits, Links)
                if (
                    el.control_type
                    in ["Button", "Edit", "Document", "Hyperlink", "ListItem"]
                    or el.name
                ):
                    label = el.name if el.name else "Unlabeled"
                    system_desc += f"    - [{el.control_type}] Name:'{label}' ID:{el.automation_id}\n"
                    count += 1
        system_desc += "\n"

    # Gemma 3 Standard Chat Format
    final_prompt = f"<start_of_turn>user\nCONTEXT:\n{system_desc}\n\nUSER COMMAND: {user_query}<end_of_turn>\n<start_of_turn>model\n"
    return final_prompt


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    elif size < 1024**2:
        return f"{size / 1024:.2f} KB"
    else:
        return f"{size / 1024**2:.2f} MB"


# ==========================================
# 4. WEBSOCKET SERVER
# ==========================================


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global CURRENT_CONTEXT
    await websocket.accept()
    print("[Server] Client Connected via WebSocket")

    try:
        while True:
            # We accept either Text (Commands) or Bytes (Compressed State)
            message = await websocket.receive()

            payload_data = None

            # 1. HANDLE BYTES (Compressed State Update)
            if "bytes" in message and message["bytes"]:
                compressed_data = message["bytes"]
                try:
                    # Decompress Zstd
                    with dctx.stream_reader(compressed_data) as reader:
                        decompressed_data = reader.read()

                    # Parse JSON
                    json_data = json.loads(decompressed_data)
                    payload_data = json_data

                    # Debug log
                    print(
                        f"[Sync] Recv: {format_bytes(len(compressed_data))} -> Decompressed: {format_bytes(len(decompressed_data))}"
                    )

                except Exception as e:
                    print(f"[Error] Decompression failed: {e}")
                    continue

            # 2. HANDLE TEXT (Uncompressed JSON)
            elif "text" in message and message["text"]:
                try:
                    payload_data = json.loads(message["text"])
                except:
                    print("[Error] Invalid JSON text")
                    continue

            # 3. PROCESS PAYLOAD
            if payload_data:
                msg_type = payload_data.get("type")

                # CASE A: UPDATE STATE
                if msg_type == "state_update":
                    if "data" in payload_data:
                        try:
                            # Update Global State
                            CURRENT_CONTEXT = SystemContext(**payload_data["data"])
                            # print(f"[State] Updated. Active Windows: {len(CURRENT_CONTEXT.windows)}")
                        except Exception as e:
                            print(f"[Error] Pydantic Validation Failed: {e}")

                # CASE B: PROMPT / QUESTION
                elif msg_type == "prompt":
                    user_text = payload_data.get("text", "")
                    print(f"[Prompt] User: {user_text}")

                    if llm:
                        # Build prompt using the FRESH context
                        full_prompt = build_prompt(user_text, CURRENT_CONTEXT)

                        # Stream Output
                        stream = llm(
                            full_prompt,
                            max_tokens=1024,
                            stop=["<end_of_turn>"],
                            stream=True,
                            temperature=0.7,
                        )

                        for output in stream:
                            chunk = output["choices"][0]["text"]
                            await websocket.send_text(chunk)

                        await websocket.send_text("[DONE]")
                    else:
                        await websocket.send_text(
                            "Error: Model loading... try again in 10s.[DONE]"
                        )

    except WebSocketDisconnect:
        print("[Server] Client Disconnected")
    except Exception as e:
        print(f"[Server] Crash: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8384)

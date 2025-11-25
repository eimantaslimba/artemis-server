import uvicorn
import json
import zstandard
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from llama_cpp import Llama
from pydantic import BaseModel
from typing import List, Dict, Optional


# --- (Pydantic Models are unchanged) ---
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


# --- (Global State & Model are unchanged) ---
CURRENT_CONTEXT = SystemContext()
dctx = zstandard.ZstdDecompressor()
MODEL_PATH = "/workspace/model.gguf"
print("--- INITIALIZING GEMMA 3 27B ---")
try:
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=8192, verbose=False)
    print("--- MODEL LOADED ---")
except Exception as e:
    print(f"!!! MODEL LOAD FAILED: {e}")
    llm = None
app = FastAPI()


# --- (build_prompt and format_bytes are unchanged) ---
def build_prompt(user_query: str, context: SystemContext) -> str:
    system_desc = "Current System UI State:\n"
    if not context.windows:
        system_desc += "No active windows detected.\n"
    for i, win in enumerate(context.windows):
        info = win.info
        system_desc += f"Window {i}: '{info.title}' (App: {info.class_name})\n"
        if win.contents:
            system_desc += "  Key UI Elements:\n"
            count = 0
            for el_id, el in win.contents.elements.items():
                if count > 30:
                    system_desc += "  ... (more elements)\n"
                    break
                if (
                    el.control_type
                    in ["Button", "Edit", "Document", "Hyperlink", "ListItem"]
                    or el.name
                ):
                    label = el.name if el.name else "Unlabeled"
                    system_desc += f"    - [{el.control_type}] Name:'{label}' ID:{el.automation_id}\n"
                    count += 1
        system_desc += "\n"
    return f"<start_of_turn>user\nCONTEXT:\n{system_desc}\n\nUSER COMMAND: {user_query}<end_of_turn>\n<start_of_turn>model\n"


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    elif size < 1024**2:
        return f"{size / 1024:.2f} KB"
    else:
        return f"{size / 1024**2:.2f} MB"


# ==========================================
# 4. WEBSOCKET SERVER (BYTES-ONLY)
# ==========================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global CURRENT_CONTEXT
    await websocket.accept()
    print("[Server] Client Connected (Bytes-Only Mode)")
    try:
        while True:
            # We ONLY accept binary data now.
            compressed_data = await websocket.receive_bytes()

            try:
                with dctx.stream_reader(compressed_data) as reader:
                    decompressed_data = reader.read()
                payload_data = json.loads(decompressed_data)
            except Exception as e:
                print(f"[Error] Decompression/JSON parse failed: {e}")
                continue

            msg_type = payload_data.get("type")

            # CASE A: UPDATE STATE
            if msg_type == "state_update":
                try:
                    CURRENT_CONTEXT = SystemContext(**payload_data["data"])
                    # print(f"[State] Updated. Windows: {len(CURRENT_CONTEXT.windows)}")
                except Exception as e:
                    print(f"[Error] Pydantic Validation Failed: {e}")

            # CASE B: PROMPT
            elif msg_type == "prompt":
                user_text = payload_data.get("text", "")
                print(f"[Prompt] User: {user_text}")
                if llm:
                    full_prompt = build_prompt(user_text, CURRENT_CONTEXT)
                    stream = llm(
                        full_prompt,
                        max_tokens=1024,
                        stop=["<end_of_turn>"],
                        stream=True,
                    )
                    for output in stream:
                        chunk = output["choices"][0]["text"]
                        await websocket.send_text(chunk)
                    await websocket.send_text("[DONE]")
    except WebSocketDisconnect:
        print("[Server] Client Disconnected")
    except Exception as e:
        print(f"[Server] Crash: {e}")


if __name__ == "__main__":
    # Using the port from your Cloudflare Tunnel
    uvicorn.run(app, host="0.0.0.0", port=35540)

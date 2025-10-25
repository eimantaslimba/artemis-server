# server.py

import os
import base64
import io
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional

# --- Correct Imports from the New SDK Documentation ---
from google import genai
from google.genai import types
from PIL import Image

# --- Configuration ---
load_dotenv()
if not os.getenv("GEMINI_API_key"):
    raise ValueError("GEMINI_API_KEY is not set in the .env file.")

client = genai.Client()

# ===================================================================================
# --- Detailed Pydantic Models for Strict API Validation (Corrected) ---
# ===================================================================================


# --- Models for 'contents' array ---
class InlineData(BaseModel):
    mime_type: str
    data: str


class FunctionCall(BaseModel):
    name: str
    args: Dict[str, Any]


class FunctionResponse(BaseModel):
    name: str
    response: Dict[str, Any]


class Part(BaseModel):
    text: Optional[str] = None
    inline_data: Optional[InlineData] = None
    function_call: Optional[FunctionCall] = None
    function_response: Optional[FunctionResponse] = None


class Content(BaseModel):
    role: Literal["user", "model", "function"]
    parts: List[Part]


# --- Models for 'generation_config' ---


class FunctionParameters(BaseModel):
    type: Literal["object"]
    properties: Dict[str, Any]
    required: Optional[List[str]] = None


class FunctionDeclaration(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters


class Tool(BaseModel):
    function_declarations: List[FunctionDeclaration]


# THIS IS THE NEW, CORRECTED PART
class ThinkingConfig(BaseModel):
    thinking_budget: Optional[int] = Field(
        None, description="Token budget for thinking. 0 disables, -1 enables dynamic."
    )
    include_thoughts: Optional[bool] = Field(
        None, description="If true, includes thought summaries in the response."
    )


class GenerationConfig(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[Tool]] = None
    thinking_config: Optional[ThinkingConfig] = None  # ADDED THIS LINE


# --- The Top-Level Request Model ---
class PromptRequest(BaseModel):
    user_token: str
    model_name: str
    contents: List[Content]
    generation_config: Optional[GenerationConfig] = None
    system_instruction: Optional[str] = None


# ===================================================================================

app = FastAPI()


# --- Placeholder Functions ---
def is_user_authenticated(token: str) -> bool:
    print(f"Authenticating token: {token[:10]}...")
    return True


def check_and_deduct_credits(token: str) -> bool:
    print(f"Checking credits for user with token: {token[:10]}...")
    return True


# --- The Main API Endpoint ---
@app.post("/api/v1/generate")
async def generate_response(request: PromptRequest):
    if not is_user_authenticated(request.user_token):
        raise HTTPException(status_code=401, detail="Invalid user token")
    if not check_and_deduct_credits(request.user_token):
        raise HTTPException(status_code=402, detail="Insufficient credits")

    try:
        sdk_contents = []
        for content in request.contents:
            for part in content.parts:
                if part.text is not None:
                    sdk_contents.append(part.text)
                elif part.inline_data is not None:
                    image_bytes = base64.b64decode(part.inline_data.data)
                    image = Image.open(io.BytesIO(image_bytes))
                    sdk_contents.append(image)
                elif part.function_call is not None:
                    sdk_contents.append({"function_call": part.function_call.dict()})
                elif part.function_response is not None:
                    sdk_contents.append(
                        {"function_response": part.function_response.dict()}
                    )

        sdk_generation_config = (
            request.generation_config.dict(exclude_unset=True)
            if request.generation_config
            else None
        )

        async def stream_generator():
            print(f"Requesting stream from Gemini model: {request.model_name}...")
            try:
                response_stream = client.models.generate_content_stream(
                    model=request.model_name,
                    contents=sdk_contents,
                    config=sdk_generation_config,
                    system_instruction=request.system_instruction,
                )

                for chunk in response_stream:
                    if hasattr(chunk, "text") and chunk.text:
                        yield chunk.text.encode("utf-8")
                    if hasattr(chunk, "function_calls"):
                        for func_call in chunk.function_calls:
                            yield f'FUNCTION_CALL:{json.dumps({"name": func_call.name, "args": func_call.args})}'.encode(
                                "utf-8"
                            )
            except Exception as e:
                error_message = f"Error during Gemini API call: {e}"
                print(error_message)
                yield error_message.encode("utf-8")

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
        print(f"An unexpected server error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

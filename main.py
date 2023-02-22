"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse

import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None

def input_aggregator(question, metadata):
    template = f"{question}"
    if metadata["topic"] != "":
        template += f"with topic: {metadata['topic']}"
    if metadata["context"] != "":
        template += f"context of the question {metadata['context']}"
    if metadata["legalAct"] != "":
        template += f"in legal act: {metadata['legalAct']}"
    if metadata["article"] != "":
        template += f"in article: {metadata['article']}"
    if metadata["comDelegated"] != "":
        template += f"with com delegated: {metadata['comDelegated']}"
    return template


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)
    with open("sample_vectorstore.pkl", "rb") as k:
        global sample_vectorstore
        sample_vectorstore = pickle.load(k)


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    qa_chain = get_chain(vectorstore, sample_vectorstore, question_handler, stream_handler)

    while True:
        try:
            # Receive and send back the client message
            data = await websocket.receive_text()
            data = json.loads(data)
            question = data["message"]
            metadata = {}
            for key in data.keys():
                if key != "message":
                    metadata[key] = data[key]
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": input_aggregator(question, metadata)}
            )

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)

import os

import uvicorn
from agents import set_trace_processors
from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor

from dotenv import load_dotenv

load_dotenv()


def main():
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    set_trace_processors([OpenAIAgentsTracingProcessor()])
    main()

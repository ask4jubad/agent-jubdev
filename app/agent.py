# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import google.auth
from google.adk.agents import Agent, SequentialAgent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.tools import google_search
from app.corrector import corrector_agent
from typing import AsyncGenerator

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
# os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
# os.environ.setdefault("GOOGLE_API_KEY", "YOUR_API_KEY")
# os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "False")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


class SaveQueryAgent(BaseAgent):
    """A simple agent to save the initial user query to the session state."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        if ctx.session.events:
            user_query = ctx.session.events[-1].content.parts[0].text
            ctx.session.state["user_query"] = user_query
        yield Event(author=self.name)


primary_agent = Agent(
    name="ethical_content_reviewer",
    model="gemini-2.5-flash",
    instruction="You are an AI assistant for content creators. Your purpose is to review draft text (or descriptions of images/videos) for potential ethical concerns, such as accidental bias, misinformation, privacy violations, or problematic tones. Identify these issues and suggest constructive ways to rephrase or mitigate the risks.",
    tools=[google_search],
    output_key="primary_response",
)

root_agent = SequentialAgent(
    name="self_correcting_reviewer",
    sub_agents=[
        SaveQueryAgent(name="save_query"),
        primary_agent,
        corrector_agent,
    ],
    description="A primary agent that reviews content for ethical concerns, with its output then refined by a corrector agent.",
)

# Copyright 2023 LiveKit, Inc.
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

"""AssemblyAI streaming speech-to-text plugin for LiveKit Agents.

This is a local copy of `livekit.plugins.assemblyai`, vendored so we can add the
`universal-3-7-preview` model (the 3.7 Realtime Agent) to the model list and to
`_U3_PRO_MODELS`, which gates the Pro options (keyterms_prompt, agent_context,
and others). The published plugin only allows those options for a hardcoded set
of models and does not list the preview model.

Example:
    from assemblyai_stt import STT

    session = AgentSession(stt=STT(model="universal-3-7-preview"))
"""

from .stt import STT, SpeechStream
from .version import __version__

__all__ = [
    "STT",
    "SpeechStream",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class AssemblyAISTTPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(AssemblyAISTTPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False

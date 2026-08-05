# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Turning one finished rollout record into a token-bearing one.

This is the step between capture and training: the rollout has run, its model
calls are in the store, and the record needs its ``response.output`` replaced
with the merged trajectory before anything reads it.

It is a module of its own, rather than a block inside the rollout-collection
loop, because that loop is not the only caller. A process that drives
``run_examples`` itself gets the finished records without ever entering
``run_from_config``, and would otherwise have to reimplement the gating, the id
derivation, the build, the metrics and the retention to get the same records.
One function means the two cannot drift.

This is also the layer that knows the shape of a rollout record, which is why it
sits outside the package's leaf export surface: reading a record's agent
reference and its correlation key means importing Gym's config, and that pulls
in the server stack. ``consumer.py`` below it stays free of both, so a caller
with its own record format can use the build without inheriting Gym's.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    TOKEN_ID_CAPTURE_KEY_NAME,
    get_first_server_config_dict,
    get_global_config_dict,
)
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.token_id_capture.consumer import (
    token_id_capture_dirs_from_config,
    trajectories_for_rollout,
)
from nemo_gym.token_id_capture.store import TokenCaptureStore


# Per-rollout token-capture health, attached to the record so a run can see what the build dropped:
# chain count, quarantined fraction, delivered token fraction, and whether the rollout should be
# masked (an incomplete capture, or a final-call retry that cannot be resolved).
TOKEN_CAPTURE_KEY = "_ng_token_capture"

# Top-level flag asking a consumer to drop this rollout from the loss. The detail sits under
# TOKEN_CAPTURE_KEY; this is the one field a consumer has to read, and it is the name already used
# for the same request elsewhere in the repo.
MASK_SAMPLE_KEY = "mask_sample"


def gym_reads_back_tokens(global_config: Any) -> bool:
    """Whether ``finalize_rollout_token_capture`` can do anything for this run.

    False when capture is off, when no store directory resolves, or when delivery is external and
    the caller reads the records through its own ``TokenSource``. Exposed so the rollout-collection
    loop can answer this once instead of per record.
    """
    if not token_id_capture_dirs_from_config(global_config):
        return False
    return TokenIdCaptureConfig.model_validate(global_config).token_id_capture_delivery == "response"


def agent_participates_in_token_capture(global_config: Any, agent_name: str | None) -> bool:
    """Whether the producing agent opted into training token capture (its ``token_id_capture``
    flag). Native agents leave it off, since they carry token ids inline and need no store rebuild,
    so this scopes the rebuild (and its token-less warnings) to external-harness rollouts. Defaults
    to False when the agent or flag is absent."""
    if not agent_name:
        return False
    try:
        agent_config = get_first_server_config_dict(global_config, agent_name)
    except (KeyError, IndexError, TypeError):
        return False
    return bool((agent_config or {}).get(TOKEN_ID_CAPTURE_KEY_NAME, False))


def finalize_rollout_token_capture(
    result: dict,
    global_config: Any = None,
    token_capture_dirs: list[Path] | None = None,
) -> None:
    """Rebuild one finished rollout record's ``response.output`` from its captured tokens.

    Call this once per record, after the harness and verifier are done with it. Mutates ``result``
    in place. The reward, the reward components and everything else the harness and verifier
    produced are left alone; only ``response.output`` is replaced, so a trainer reads an external
    harness's rollout exactly as it reads a native agent's, with no sidecar payload and no
    per-agent branch.

    A no-op unless capture is on, delivery is Gym's, and the agent that produced the record opted
    in. Both optional arguments are resolved from the process config when omitted, so a caller with
    one record and no other context can pass just the record; a caller in a loop passes them in to
    avoid re-resolving per record.

    Never raises. A rollout whose tokens are missing or ambiguous is marked for masking, since the
    alternative is training on a trajectory with a hole in it.
    """
    if global_config is None:
        global_config = get_global_config_dict()
    if token_capture_dirs is None:
        token_capture_dirs = (
            token_id_capture_dirs_from_config(global_config) if gym_reads_back_tokens(global_config) else []
        )
    if not token_capture_dirs:
        return

    agent_name = (result.get(AGENT_REF_KEY_NAME) or {}).get("name")
    if not agent_participates_in_token_capture(global_config, agent_name):
        return

    rollout_id = maybe_rollout_id_from_run_body(result)
    if rollout_id is None:
        # The rollout carries no id and no indices, so its model calls could not have been
        # correlated. Report it rather than passing over it: a token-less rollout is an unusable
        # training sample and looks like a healthy one. Usually the agent did not apply the
        # /ng-rollout prefix.
        warnings.warn(
            "training token capture is enabled but a rollout result carries no id and no "
            "task/rollout indices; its model calls were not correlated or captured and it will be "
            "token-less. Check that the agent applies the /ng-rollout prefix.",
            stacklevel=2,
        )
        return

    response = result.get("response") if isinstance(result.get("response"), dict) else {}
    built = trajectories_for_rollout(rollout_id, token_capture_dirs, model=str(response.get("model") or ""))
    if built is None:
        # A derivable rollout id but an empty store means correlation broke somewhere between the
        # agent and the capture middleware, such as an external harness or proxy that did not
        # preserve the prefix. Surface it rather than emitting a token-less rollout into training.
        warnings.warn(
            f"training token capture is enabled but no tokens were captured for rollout "
            f"{rollout_id}; its response.output was not rebuilt and it will be token-less. The "
            "agent's model calls likely did not reach the capture middleware correlated.",
            stacklevel=2,
        )
        return

    projected = built["rebuilt_response"]
    if projected is not None:
        if isinstance(result.get("response"), dict):
            result["response"]["output"] = projected["output"]
        else:
            result["response"] = projected

    # Carry what the build dropped onto the record. Without this the quarantine and delivered-token
    # fractions are computed and discarded, and a rollout that trained on one of five calls looks
    # like one that trained on all five.
    record_metrics = dict(built.get("metrics") or {})
    if built.get("error"):
        record_metrics["error"] = built["error"]
    if built.get(MASK_SAMPLE_KEY):
        record_metrics[MASK_SAMPLE_KEY] = True
        # Also at the top level, where a consumer finds it without knowing this feature exists.
        result[MASK_SAMPLE_KEY] = True
        warnings.warn(
            f"rollout {rollout_id} was captured incompletely or ambiguously ({record_metrics}); "
            "it is marked for masking rather than trained on.",
            stacklevel=2,
        )
    result[TOKEN_CAPTURE_KEY] = record_metrics

    # Delete on consume. Rollouts write hundreds of KB each and the store appends, so keeping
    # consumed files both grows the directory without bound and lets a later run with the same id
    # append onto them. Failed builds keep their records: they are the only evidence of why the
    # build failed.
    if projected is not None and not os.environ.get("NG_KEEP_CONSUMED_TOKEN_CAPTURES"):
        for capture_dir in token_capture_dirs:
            TokenCaptureStore(capture_dir).delete(rollout_id)

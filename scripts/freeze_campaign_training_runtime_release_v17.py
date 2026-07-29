#!/usr/bin/env python3
"""Freeze Campaign Runtime V17 for repaired V25 bytes on gpu35."""

from __future__ import annotations

import freeze_campaign_training_runtime_release_v16 as runtime_v16


runtime_v16.RELEASE_ID = "TRAINING_RUNTIME_RELEASE_V17"
runtime_v16.RELEASE_RESOURCE = "training_runtime_release_v17.json"
runtime_v16.BASE_RELEASE_RESOURCE = "training_runtime_release_v16.json"
runtime_v16.LOCK_RESOURCE = "training_runtime_v17_lock.json"
runtime_v16.BACKEND_CLASS = (
    "LAB_GPU35_RTX3080_NATIVE_LINUX_V7_EFFECT_V25"
)


if __name__ == "__main__":
    runtime_v16.main()

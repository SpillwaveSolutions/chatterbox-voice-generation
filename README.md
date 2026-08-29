# chatterbox-voice-generation

Library-first Chatterbox voice synthesis: text + config + seed to WAV. Ships the
`tts-chatterbox` Python distribution (import packages `tts_chatterbox` and
`artifact_store`) plus the CPU jobs-API service and the RunPod GPU worker.

**Voice references are biometric data: cloning a voice requires the speaker's explicit
permission, and voice-reference files must never be committed to this repository or baked
into its images.** This library is not a consent-free voice-cloning tool. See
`docs/CONSUMPTION.md` for the full consumption contract.

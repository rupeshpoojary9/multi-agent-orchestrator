"""In-process mock backends the foundational agents drive.

`api_backend` is a mock enterprise REST service over SQLite (the 'modern'
system). `web_app` is a mock legacy CRUD web UI modeled as a small state machine
(the 'legacy' system the Web Agent clicks through). Cross-system tasks read from
one and act in the other, exactly like the real orchestrator's remit.
"""

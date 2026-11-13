
# ============================================================
# KubeAssist - Guardrails Configuration
# ============================================================
#
# Request classification is now handled by:
#   Portkey + Groq
#
# NeMo semantic input flows are no longer used for
# request classification.
#
# Jina embeddings remain dedicated to the RAG pipeline.
#
# ============================================================

PASS_THROUGH_TOKEN = "KUBEASSIST_ALLOW_REQUEST"

# Retained as empty compatibility constants in case another
# project module still imports these names.
# They are not used by the new classifier.

COLANG_CONTENT = ""

YAML_CONTENT = ""

RAIL_INDICATORS = []

import sys
sys.path.insert(0, ".")
from config import MODEL_NAME
from llm_client import _extract_retry_delay

print("Model:", MODEL_NAME)

# Test retryDelay parser
delay1 = _extract_retry_delay(Exception("Please retry in 45.1s"))
print("retryDelay from plain text:", delay1, "s")

delay2 = _extract_retry_delay(Exception("'retryDelay': '48s'"))
print("retryDelay from error dict:", delay2, "s")

delay3 = _extract_retry_delay(Exception("some other error"))
print("retryDelay from unrelated error:", delay3)

print("ALL OK")

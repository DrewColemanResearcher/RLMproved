from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

# Remove the script directory so `import openai` resolves to the installed SDK,
# not `rlm/clients/openai.py` next to this smoke test.
sys.path = [p for p in sys.path if Path(p).resolve() != SCRIPT_DIR]

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from rlm.clients.aruba_client import ArubaClientChatCompletion, ArubaClientResponses


def main() -> None:
	prompt = [{"role": "user", "content": "Ciao come va?"}]

	print("=== Aruba chat completions (legacy) ===")
	chat_client = ArubaClientChatCompletion(
		model_name="openai/Qwen2.5-VL-7B-Instruct:demo",
		method="legacy",
	)
	print(chat_client.completion(prompt=prompt))

	print("\n=== Aruba responses (legacy) ===")
	responses_client = ArubaClientResponses(
		model_name="openai/gpt-oss-20b:demo-30k",
		method="legacy",
	)
	print(responses_client.completion(prompt=prompt))


if __name__ == "__main__":
	main()

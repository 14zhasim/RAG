import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

def main() -> None:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    model = "openrouter/free"
    messages = [
        {
            "role": "user",
            "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    if response.usage is None:
        raise RuntimeError("API response has no usage data")

    print(response.choices[0].message.content)
    print("Prompt tokens: ", response.usage.prompt_tokens)
    print("Response tokens: ", response.usage.completion_tokens)

if __name__ == "__main__":
    main()
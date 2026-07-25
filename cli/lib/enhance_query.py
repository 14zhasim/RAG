import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
model = "openrouter/free"


def get_client():
    # Build the API client lazily so commands that do not use query enhancement
    # can run without requiring OPENROUTER_API_KEY.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """
    message = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=message,
    )

    if response.usage is None:
        raise RuntimeError("API response has no usage data")
    
    #defensive programming here - in case llm does not return valid keyword, return original
    #and also stripping LLM response of things like whitespace, speech marks etc.!
    corrected = (response.choices[0].message.content or "").strip().strip('"')
    return corrected if corrected else query

#make sure the method here is optional, default value = None
def enhance_query(query: str, method: str | None = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case _:
            return query

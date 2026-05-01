from llmproxy import LLMProxy
import re
import json
from pathlib import Path
from time import sleep
from string import Template


def rag_context_string_simple(rag_context):
    """
    Input
    1. 

    Output:
    . 
    """
    if isinstance(rag_context, str):
        return rag_context
    return "\n".join(str(chunk) for chunk in rag_context)


def build_batch_query(batch_start: int, batch_size: int, total: int) -> str:
    """
    Returns the prompt for a single batch.
    Complexity range is calculated proportionally from the batch's position.
    """
    # Derive complexity range from where this batch sits in the full sequence
    batch_end = batch_start + batch_size - 1
    
    progress_start = (batch_start - 1) / total 
    progress_end   = (batch_start - 1 + batch_size) / total

    complexity_min = max(1, round(progress_start * 10 + 1))
    complexity_max = min(10, round(progress_end   * 10))

    # Ensure min never exceeds max (edge case on last batch)
    if complexity_min > complexity_max:
        complexity_min = complexity_max

    return f"""
    You have to create exactly {batch_size} Latin language exercises,
    numbered {batch_start} through {batch_start + batch_size - 1} (part of {total} total).

    COMPLEXITY CONSTRAINT (mandatory):
        Every exercise in this batch must have a complexity between {complexity_min} and {complexity_max}.
        Spread the values across that range — do not repeat the same complexity for all exercises.
        Use the RAG context as a reference for what each level looks like in practice.

    Divide them equally among the three types: vocabulary, grammar, main_verb.
    Do NOT use any exercise from any textbook. Only ancient/classical Roman authors.

    Return ONLY a JSON array — no preamble, no markdown fences, no trailing text.
    The array must be properly closed with ].

    STRUCTURES:

    1. VOCABULARY — Complete the sentence
        Fill a blank with the correct word.
        Example: "Dionysius tum erat ________(tyrant) Syracusanorum."
        Answer: "tyrannus"
        Rules:
            - One blank per sentence, marked as ________.
            - English meaning of missing word in parentheses.
            - Answer must be unambiguous.

    2. GRAMMAR — Multiple choice (single option)
        Select the correct option to complete or fix a sentence.
        Example: "De mortuis nihil nisi bonum ________."
                  a) dixo  b) dicimur  c) dicamus  d) dictum  → c) dicamus
        Rules:
            - Exactly 4 options (a, b, c, d). Only one correct.
            - Distractors must be plausible but clearly wrong.

    3. MAIN VERB — True or False
        Is the bracketed word the main verb?
        Example: "Morti Socratis semper illacrimo, [legens] Platonem." → False
        Rules:
            - Exactly one word bracketed with [].
            - Sentence must be unambiguous.

    COMPLEXITY LEVELS (1–10):
        1–3  → Simple vocabulary, present/past tense, short sentences (A1–A2)
        4–6  → Compound sentences, varied tenses, academic vocabulary (B1–B2)
        7–10 → Complex clauses, technical/literary vocabulary, nuanced grammar (C1–C2)
        Use the RAG context as a reference for the complexity scale.

    SOURCE: Draw from the Latin Corpus or PHI only.
    Cite title, author, book, and line.

    Schema for each exercise in the array:
    {{
        "exercise": <number from {batch_start} to {batch_end}>,
        "type": "vocabulary" | "grammar" | "main_verb",
        "complexity": <1–10>,
        "source": {{
            "title": "...",
            "reference": "author/book/line"
        }},
        "sentence": "...",
        "question": "..." | null,
        "options": ["a) ...", "b) ...", "c) ...", "d) ..."] | null,
        "answer": "...",
        "answer_justification": "..."
    }}

    Notes:
        - "options" is only populated for grammar exercises, null otherwise.
        - "answer_justification" ≤ 15 words. Abbreviate if needed.
    """


def extract_json_array(raw: str) -> list:
    """
    Strips fences and extracts the JSON array from a raw model response.
    Raises ValueError if no valid array is found.

    Input:
    1. raw: str that contains the json information.

    Output:
    2. list containing the cleaned json information.
    """
    cleaned = raw.strip().lstrip("`'\"").rstrip("`'\"")
    match = re.search(r'(\[.*\])', cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in response:\n{raw[:300]}")
    return json.loads(match.group(1))


def generate_exercises(
    client,
    total: int = 30,
    batch_size: int = 5,
    rag_session: str = "RAG",
    gen_session: str = "GeneratorSession",
    rag_threshold: float = 0.6,
    rag_k: int = 4,
    model: str = "gemini-2.5-flash-lite",
    temperature: float = 0.5,
    output_path: str = "exercises.json",
    retry_attempts: int = 2,
    delay_between_batches: float = 1.0) -> list:
    
    """
    Generates 'total' exercises in batches of 'batch_size'.

    Each batch is an independent API call, so the model never has to produce
    a large JSON in one shot — preventing mid-output truncation.

    Input:
    
    1. client: LLMProxy instance (already configured with RAG docs).
    2. total: int that shows the total number of exercises to generate.
    3. batch_size: int, How many exercises per API call (5 is a safe default).
    4. rag_session: str, the session ID where RAG documents were uploaded.
    5. gen_session: str, the session ID used for generation calls.
    6. rag_threshold: float, the minimum similarity score for RAG retrieval.
    7. rag_k: int, number of RAG chunks to retrieve.
    8. model: str, the LLM Model.
    9. temperature: int, sampling temperature.
    10. output_path: str, the location to write the final merged JSON file.
    11. retry_attempts: int, how many times to retry a failed batch before skipping.
    12. delay_between_batches: float, amount of seconds to wait between API calls (rate-limit safety).

    Output
    
    1. List of all exercise dicts, merged in order.
    """

    all_exercises: list = []
    batch_number = 0

    batch_start = 1
    while batch_start <= total:
    # Last batch may be smaller than batch_size
        current_batch_size = min(batch_size, total - batch_start + 1)
        batch_number += 1

        print(f"[Batch {batch_number}] Exercises {batch_start}–"
              f"{batch_start + current_batch_size - 1} ...", end=" ", flush=True)
        
        query = build_batch_query(batch_start, current_batch_size, total)

    # RAG retrieval (same pattern as original code)
        rag_context = client.retrieve(
            query=query,
            session_id=rag_session,
            rag_threshold=rag_threshold,
            rag_k=rag_k,
        )
        query_with_rag = Template("$query\n$rag_context").substitute(
            query=query,
            rag_context=rag_context_string_simple(rag_context),
        )

    # Generation with retry logic
        batch_exercises = None
        for attempt in range(1, retry_attempts + 1):
            try:
                response = client.generate(
                    model=model,
                    system=(
                        "You are a professor creating Latin language exercises for students. "
                        "Return ONLY a valid JSON array. No markdown, no explanation."
                    ),
                    query=query_with_rag,
                    temperature=temperature,
                    lastk=0,         
                    session_id=gen_session,
                    rag_usage=True,
                )

                batch_exercises = extract_json_array(response["result"])

    # Validate exercise count
                if len(batch_exercises) != current_batch_size:
                    raise ValueError(
                        f"Expected {current_batch_size} exercises, "
                        f"got {len(batch_exercises)}"
                    )

    # Validate exercise numbers
                expected_nums = set(range(batch_start, batch_start + current_batch_size))
                actual_nums   = {ex.get("exercise") for ex in batch_exercises}
                if expected_nums != actual_nums:
    # Non-fatal: re-number to maintain sequence integrity
                    print(f"[warn] numbering mismatch, re-numbering...", end=" ")
                    for i, ex in enumerate(batch_exercises):
                        ex["exercise"] = batch_start + i

                print(f"OK ({len(batch_exercises)} exercises)")
                break

            except (ValueError, json.JSONDecodeError, KeyError) as e:
                print(f"attempt {attempt} failed: {e}", end=" ")
                if attempt < retry_attempts:
                    sleep(2)
                else:
                    print(f"— skipping batch {batch_number}")
    # Insert placeholders so numbering stays consistent downstream
                    batch_exercises = [
                        {
                            "exercise": batch_start + i,
                            "type": "ERROR",
                            "error": str(e),
                        }
                        for i in range(current_batch_size)
                    ]

        all_exercises.extend(batch_exercises)
        batch_start += current_batch_size

        if batch_start <= total:
            sleep(delay_between_batches)

    #  Write merged output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_exercises, f, ensure_ascii=False, indent=2)

    print(f"\nDone. {len(all_exercises)} exercises written to '{output_path}'.")
    return all_exercises


if __name__ == "__main__":

    client = LLMProxy()

    # Upload RAG documents (same as original)
    base_dir = Path(__file__).parent
    address_rag = base_dir / "RAG" / "RAG_Score.json"

    with open(address_rag, mode="r", encoding="utf-8") as y:
        rag_context = y.read()

    client.upload_text(
        text=rag_context,
        session_id="RAG",
        strategy="fixed",
    )
    
    print("Thinking...")
    sleep(20) 

    # Generate in batches
    exercises = generate_exercises(
        client=client,
        total=30,
        batch_size=5,
        output_path="exercises.json",
    )
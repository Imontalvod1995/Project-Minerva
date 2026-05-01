
import json
from llmproxy import LLMProxy
from pathlib import Path
from string import Template
import re
from time import sleep

def rag_context_string_simple(rag_context):
    """
    Convert the RAG context list (from retrieve API)
    into a single plain-text string that can be appended to a query.
    """
    context_string = ""
    i = 1
    for collection in rag_context:
        if not context_string:
            context_string = "The following is additional context that may be helpful in answering the user's query."
 
        context_string += """
        #{} {}
        """.format(i, collection['doc_summary'])
        j = 1
        for chunk in collection['chunks']:
            context_string += """
            #{}.{} {}
            """.format(i, j, chunk)
            j += 1
        i += 1
    return context_string
 
 
CORRECTOR_SYSTEM = """
    You are a Latin language expert evaluating and correcting exercises.
    You will receive a list of exercises, each accompanied by the verdicts
    from three jurors (Melchior, Casper, and Balthazar). Each juror verdict
    follows this structure:
 
    {
        "exercise_id": int,
        "verdict": "APPROVE" | "FLAG" | "REJECT",
        "confidence": 0-1,
        "priority": 1-5,
        "tags": ["..."],
        "summary": "...",
        "errors": [
            { "type": "conceptual"|"factual"|"formatting",
          "severity": "critical"|"minor",
          "description": "..." }
            ]
    }
 
    Follow these rules when processing each exercise:
 
        1. APPROVED exercises: If all jurors approved and no errors were reported,
            return the exercise unchanged using the output schema below.
 
        2. FLAGGED or REJECTED exercises: Apply the corrections supported by the
            majority of jurors. If all three jurors disagree with each other,
            do not attempt a correction — instead return the exercise with
            "needs_review": true and leave all other fields as received.
 
        3. All sentences must be drawn from the Latin Corpus or the PHI.
            You must cite the source title, author, book, and line number.
 
        4. Complexity scale:
            1-2: Basic vocabulary, simple present tense, single clause.
            3-4: Introductory grammar, common verb forms, simple sentences.
            5-6: Intermediate grammar, subordinate clauses, less common vocabulary.
            7-8: Advanced syntax, participles, indirect speech, rare vocabulary.
            9-10: Expert level, complex periodic sentences, archaic or poetic forms.
 
    Return a JSON array with one object per exercise using this schema:
        {
            "exercise_id": int,
            "needs_review": true | false,
            "type": "vocabulary" | "grammar" | "main_verb",
            "complexity": 1-10,
            "source": {
                        "title": "...",
                        "author": "...",
                        "book": "...",
                        "line": "..."
                    },
            "sentence": "...",
            "question": "...",
            "options": ["a) ...", "b) ...", "c) ...", "d) ..."] | null,
            "answer": "...",
            "answer_justification": "..."
        }
 
        Notes:
            - "options" is only populated for grammar exercises, null otherwise.
            - "answer_justification" must explain why the answer is correct and,
                for grammar and main_verb exercises, why the wrong options are incorrect.
            - Return ONLY the raw JSON array. No preamble, no markdown fences,
                no explanatory text after the array.
"""
 
 
def correct_batch(client, batch_exercises, rag_context):
    """
    Send one batch of exercises to the corrector model and return
    the parsed list of corrected exercise dicts.
    """
    ids = [ex.get('exercise', ex.get('exercise_id', '?')) for ex in batch_exercises]
    print(f"  Correcting exercises {ids[0]}-{ids[-1]}...")
 
    # Build the query for this batch
    batch_json = json.dumps(batch_exercises, indent=2, ensure_ascii=False)
    query_rag = (
        "Compare the following exercises with the results from the judges and "
        "correct them:\n" + batch_json
    )
 
    # Append RAG context (juror opinions retrieved from RAG_corrector session)
    query_with_rag_context = query_rag + "\n" + rag_context_string_simple(rag_context)
 
    response = client.generate(
        model="us.anthropic.claude-3-haiku-20240307-v1:0",
        system=CORRECTOR_SYSTEM,
        query=query_with_rag_context,
        temperature=0.5,
        lastk=1,
        session_id='Corrector_Session',
        rag_usage=True
    )
 
    # The result is already a string from the model
    jstring = response["result"]
 
    # Guard: detect proxy-level error messages before attempting to parse
    ERROR_PHRASES = ["an error was encountered", "error was encountered", "internal server error"]
    if any(phrase in jstring.lower() for phrase in ERROR_PHRASES):
        raise RuntimeError(
            f"LLMProxy returned an error for exercises {ids[0]}-{ids[-1]}.\n"
            f"Full response: {jstring}\n"
            f"Possible causes: query too long, model unavailable, or RAG session expired."
        )
 
    # Strip surrounding whitespace and any accidental quotes
    jstring = jstring.strip().lstrip("`\"").rstrip("`\"'")
 
    # Remove markdown fences if present
    jstring = re.sub(r'^```(?:json)?\s*', '', jstring)
    jstring = re.sub(r'\s*```$', '', jstring)
 
    # Extract the JSON array or object
    match = re.search(r'(\[.*\]|\{.*\})', jstring, re.DOTALL)
    if not match:
        raise ValueError(
            f"No JSON found in response for exercises {ids[0]}-{ids[-1]}.\n"
            f"Raw response: {jstring[:300]}"
        )
 
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON parsing failed for exercises {ids[0]}-{ids[-1]} — "
            f"likely truncated (max_tokens too low).\n"
            f"Error: {e}\n"
            f"Last 200 chars: {jstring[-200:]}"
        ) from e
 
    # If the model returned a single object instead of a list, wrap it
    if isinstance(parsed, dict):
        parsed = [parsed]
 
    print(f"  OK — corrected {len(parsed)} exercises.")
    return parsed
 
 
if __name__ == '__main__':
 
    client = LLMProxy()
 
    base_dir = Path(__file__).parent
    rag_exercises_address = base_dir / "RAG" / "exercises.json"
    rag_judgment_address  = base_dir / "MAGI_verdict.json"
 

    with open(rag_exercises_address, mode="r", encoding="utf-8") as ex:
        all_exercises = json.load(ex)
 
    print(f"Loaded {len(all_exercises)} exercises.")
 
    with open(rag_judgment_address, mode='r', encoding='utf-8') as ju:
        context_ju = ju.read()
 
    client.upload_text(context_ju, session_id="RAG_corrector", strategy="fixed")
 
    print("Waiting for RAG indexing...")
    sleep(20)
 
    retrieval_query = (
        "Compare the following exercises with the results from the judges and correct them."
    )
 
    rag_context = client.retrieve(
        query=retrieval_query,
        session_id='RAG_corrector',
        rag_threshold=0.6,
        rag_k=4
    )
 
    BATCH_SIZE = 5
    batches = [
        all_exercises[i : i + BATCH_SIZE]
        for i in range(0, len(all_exercises), BATCH_SIZE)
    ]
 
    all_corrected = []
 
    for i, batch in enumerate(batches):
        ids = [ex.get('exercise', ex.get('exercise_id', '?')) for ex in batch]
        print(f"\nBatch {i + 1}/{len(batches)} (exercises {ids[0]}-{ids[-1]}):")
 
        # Retry up to 3 times on transient proxy errors
        MAX_RETRIES = 3
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                corrected_batch = correct_batch(client, batch, rag_context)
                all_corrected.extend(corrected_batch)
                break
            except RuntimeError as e:
                print(f"  Attempt {attempt} failed: {e}")
                if attempt < MAX_RETRIES:
                    print(f"  Retrying in 10 seconds...")
                    sleep(10)
                else:
                    print(f"  All {MAX_RETRIES} attempts failed. Skipping batch and marking for review.")
                    for ex in batch:
                        ex["needs_review"] = True
                        all_corrected.append(ex)
 
        # Brief pause between batches to avoid rate limiting
        if i < len(batches) - 1:
            print("  Pausing before next batch...")
            sleep(5)
 
    print(f"\nTotal exercises corrected: {len(all_corrected)}")
 
    if len(all_corrected) != len(all_exercises):
        print(
            f"WARNING: Expected {len(all_exercises)} exercises, "
            f"got {len(all_corrected)}."
        )
 
    needs_review = [
        ex for ex in all_corrected if ex.get("needs_review") is True
    ]
    if needs_review:
        review_ids = [ex.get("exercise_id", "?") for ex in needs_review]
        print(f"Exercises flagged for manual review: {review_ids}")
 
    output_path = base_dir / "exercises_corrected.json"
    with open(output_path, "w", encoding="utf-8") as c:
        c.write(json.dumps(all_corrected, indent=2, ensure_ascii=False))
 
    print(f"Saved to {output_path}")
 

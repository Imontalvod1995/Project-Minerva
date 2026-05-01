from llmproxy import LLMProxy
from string import Template
import json

def rag_context_string_from_exercises(exercises: list[dict]) -> str:
    """
    Convert a list of exercise dicts (loaded from the JSON file)
    into a plain-text RAG context string.
    Each exercise becomes a numbered block with all its relevant fields.
    """
    if not exercises:
        return ""
 
    lines = [
        "The following is additional context (Latin exercises) "
        "that may be helpful in answering the user's query."
    ]
 
    for i, ex in enumerate(exercises, start=1):
        source = ex.get("source", {})
        source_str = (
            f"{source.get('author', '?')}, {source.get('title', '?')} "
            f"(Book {source.get('book', '?')}, Line {source.get('line', '?')})"
        )
        lines.append(
            f"\n#{i}  [Exercise {ex['exercise_id']} | type: {ex['type']} | "
            f"complexity: {ex['complexity']}]"
        )
        lines.append(f"    Source : {source_str}")
        lines.append(f"    Sentence: {ex.get('sentence', '')}")
 
        if ex.get("question"):
            lines.append(f"    Question: {ex['question']}")
 
        if ex.get("options"):
            lines.append(f"    Options : {' | '.join(ex['options'])}")
 
        lines.append(f"    Answer  : {ex.get('answer', '')}")
        lines.append(f"    Justification: {ex.get('answer_justification', '')}")
 
    return "\n".join(lines)
 


if __name__ == "__main__":
    # 1. Load exercises from the JSON file
    JSON_PATH = "exercises_corrected.json"
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        all_exercises: list[dict] = json.load(f)
 
    # 2. Create client
    client = LLMProxy()
 
    # 3. Setup parameters
    model_name = "us.anthropic.claude-3-haiku-20240307-v1:0"
    system_instructions = (
        "You are a strict Latin academic tutor. "
        "The user's query will include a set of reference exercises, each tagged with a complexity (1-10) and a type (vocabulary, grammar, or main_verb). "
        "You MUST follow these rules without exception:\n"
        "1. DIFFICULTY: Only generate or discuss exercises that match the exact complexity level provided by the user (1=easiest, 10=hardest). "
        "Do not simplify or increase difficulty under any circumstances.\n"
        "2. EXERCISE TYPE: There are three valid types — 'vocabulary', 'grammar', and 'main_verb'. "
        "If the user specifies a type, use only that type. "
        "If no type is specified, you may pick any one type at random for each exercise, but you must still strictly follow its format: "
        "- 'vocabulary': fill-in-the-blank for a missing Latin word. "
        "- 'grammar': multiple-choice questions testing morphology, syntax, or verb forms. "
        "- 'main_verb': True/False questions asking whether a bracketed word is the main verb of the sentence.\n"
        "3. FORMAT: Mirror the exact format of the reference exercises — include sentence, question, options (if grammar), and answer justification.\n"
        "4. SOURCE FIDELITY: Base new exercises on authentic Latin texts (Caesar, Cicero, Seneca, Tacitus, Virgil, Ovid, Sallust) at a register appropriate to the complexity level.\n"
        "5. Never mix difficulties. If the user asks for complexity 3, every exercise you produce must be complexity 3. "
        "Type can vary across exercises only if the user did not specify one — but each individual exercise must perfectly conform to its chosen type's format."
    )
    temperature_value = 0.5
    last_queries = 3
    session_id_value = "conversation"
    rag_enabled = True
 
    # 4. Interactive loop with difficulty filter
    while True:
        raw = input(
            "\nEnter difficulty (1-10) and your query, e.g.  '3 Explain the grammar', "
            "or type EXIT to stop: "
        )
 
        if raw.strip().lower() == "exit":
            break
 
        # Parse difficulty from the start of the input
        parts = raw.strip().split(maxsplit=1)
        difficulty = None
        query_prompt = raw.strip()
 
        if parts and parts[0].isdigit():
            candidate = int(parts[0])
            if 1 <= candidate <= 10:
                difficulty = candidate
                query_prompt = parts[1] if len(parts) > 1 else ""
            else:
                print("Difficulty must be between 1 and 10. Ignoring difficulty filter.")
        
        if not query_prompt:
            print("Please enter a query after the difficulty level.")
            continue
 
        # Filter exercises by complexity / difficulty
        if difficulty is not None:
            filtered = [ex for ex in all_exercises if ex["complexity"] == difficulty]
            if not filtered:
                print(
                    f"No exercises found with complexity={difficulty}. "
                    "Sending all exercises as context."
                )
                filtered = all_exercises
        else:
            filtered = all_exercises
 
        rag_context_str = rag_context_string_from_exercises(filtered)
 
        # Build the final query with RAG context — now includes explicit difficulty
        query_with_rag = Template("$query\n\nTarget complexity level: $difficulty (on a 1–10 scale).\n\n$rag_context").substitute(
            query=query_prompt,
            difficulty=difficulty if difficulty is not None else "not specified",
            rag_context=rag_context_str,
        )
 
        # Turn 1: Present the exercise WITHOUT the answer
        response = client.generate(
            model=model_name,
            system=system_instructions + (
                "\n\nIMPORTANT: Present the exercise now (sentence, question, and options if any). "
                "Do NOT reveal the answer or justification yet. "
                "End your response with \'What is your answer?\'"
            ),
            query=query_with_rag,
            temperature=temperature_value,
            lastk=last_queries,
            session_id=session_id_value,
            rag_usage=rag_enabled,
        )
        print("\n" + response["result"])
 
        # Turn 2: Wait for the user\'s answer, then evaluate 
        user_answer = input("\nYour answer: ").strip()
        if user_answer.lower() == "exit":
            break
 
        response = client.generate(
            model=model_name,
            system=system_instructions + (
                "\n\nThe user has now provided their answer. "
                "Evaluate it: state whether it is correct or incorrect, "
                "reveal the correct answer, and explain the justification."
            ),
            query=f"The user answered: {user_answer}",
            temperature=temperature_value,
            lastk=last_queries,
            session_id=session_id_value,
            rag_usage=rag_enabled,
        )
        print("\n" + response["result"])


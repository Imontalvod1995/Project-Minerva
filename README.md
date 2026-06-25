PROJECT MINERVA
================
Graduate Thesis project for MA in Digital Tools for Pre Modern Studies

Thesis : https://www.proquest.com/docview/3344286994

Author: Ivan F. Montalvo D.

The project builds a pipeline that scores the complexity of Latin
sentences, generates Latin exercises from that data, has a panel of
LLMs judge the exercises, corrects the flagged ones, and finally lets
a student practice through a simple chatbot.

DEPENDENCY NOTE
----------------
All of the LLM-facing scripts (Generator.py, Magi.py, Corrector.py,
Minerva_chatbot.py) import LLMProxy, a helper library maintained by
Tufts University for talking to multiple LLM providers through one
client. It is not part of this repo and must be installed separately:
https://github.com/Tufts-University/LLMProxy

PIPELINE ORDER
---------------
The five scripts are meant to run in this order, each one producing a
file consumed by the next:

1. Abacus.py: RAG_Score.json
2. Generator.py: exercises.json
3. Magi.py: MAGI_verdict.json
4. Corrector.py: exercises_corrected.json
5. Minerva_chatbot.py: interactive study session (no file output)

FILE-BY-FILE BREAKDOWN
------------------------

1. Abacus.py
   The "sentence complexity calculator." Runs once, offline, before any
   user interaction, to build the data later used as RAG context.

   - Reads a lemma/chapter vocabulary list (lemma_chapter.txt) and a
     folder of XML files (./HTML) containing parsed Latin sentences
     (word forms, lemmas, POS tags, dependency heads).
   - decoder(): flattens the XML into a DataFrame of words per sentence.
   - grapher(): turns each sentence's dependency structure into a graph
     (using networkx) and computes shape metrics such as diameter,
     density, average degree, and subordination/tree depth.
   - lexicon(): matches sentence vocabulary against the lemma/chapter
     list to find which curriculum chapter each word belongs to.
   - counter(): scans verb POS tags against a table of verb-form
     "grades" (present indicative, perfect subjunctive, etc.) to count
     verbs and find the hardest verb form used per sentence.
   - kahn_sorter(): combines all the above metrics into a custom
     partial-order graph and topologically sorts sentences with Kahn's
     algorithm, assigning each one a 1-10 COMPLEXITY score.
   - main(): runs the whole pipeline and writes RAG_Score.json, the
     file other scripts use as grounding data.

2. Generator.py
   Generates new Latin exercises using an LLM, grounded in the
   complexity data Abacus produced.

   - Uploads RAG_Score.json to an LLMProxy RAG session.
   - build_batch_query(): builds a prompt for a batch of exercises,
     scaling the target complexity range to match the batch's position
     in the overall sequence (so easy exercises come first).
   - Three exercise types are supported: "vocabulary" (fill-in-the-blank),
     "grammar" (4-option multiple choice), and "main_verb" (true/false
     on whether a bracketed word is the main verb).
   - generate_exercises(): calls the model in small batches (default 5
     exercises at a time) to avoid truncated JSON, retries failed
     batches, re-numbers exercises if the model miscounts, and inserts
     placeholder "ERROR" entries for batches that fail repeatedly.
   - extract_json_array(): strips markdown fences/quotes and pulls the
     JSON array out of the raw model response.
   - Writes the merged result to exercises.json.

3. Magi.py
   "LLM as Jury" — three different LLMs independently grade the
   generated exercises, then their verdicts are combined.

   - Loads exercises.json and RAG_Score.json, uploads the score data
     as RAG context shared by all three jurors.
   - Three juror methods, each with its own model and personality:
       * Melchior: GPT (4o-mini), strict, focuses on conceptual
         accuracy.
       * Casper: Claude Haiku, benevolent, focuses on whether
         difficulty matches the stated complexity.
       * Balthazar: Gemini 2.5 Flash Lite, neutral, focuses on
         whether sources are properly cited.
   - Each juror returns, per exercise, a verdict (APPROVE/FLAG/REJECT),
     a confidence score, a priority, tags, a summary, and a list of
     errors with severity.
   - juror_panel(): runs all three jurors concurrently with
     ThreadPoolExecutor.
   - deliberate(): tallies the three verdicts per exercise. Any
     critical-severity error forces a REJECT; otherwise the majority
     vote decides APPROVED / FLAGGED / REJECTED.
   - Writes the combined result to MAGI_verdict.json.

4. Corrector.py
   Applies the jury's verdicts to fix or flag exercises.

   - Loads exercises.json and MAGI_verdict.json, uploads the verdicts
     as RAG context.
   - CORRECTOR_SYSTEM prompt instructs the model: approved exercises
     pass through unchanged; flagged/rejected ones get corrected
     according to the majority of jurors; if all three jurors disagree
     completely, the exercise is left as-is but marked
     "needs_review": true instead of guessing.
   - correct_batch(): sends batches of 5 exercises (with RAG context)
     to Claude Haiku, parses the JSON response, and detects/raises on
     LLMProxy-level error messages or malformed JSON.
   - Main loop retries failed batches up to 3 times, marks exercises
     "needs_review" if all retries fail, and reports any exercises
     still flagged for manual review.
   - Writes the final result to exercises_corrected.json.

5. Minerva_chatbot.py
   The student-facing interactive tutor.

   - Loads exercises_corrected.json as its exercise bank.
   - rag_context_string_from_exercises(): formats a list of exercises
     (source, sentence, question, options, answer, justification) into
     a plain-text block to use as RAG context for the model.
   - Runs an input loop where the student types a difficulty (1-10)
     plus a request, e.g. "3 give me a vocabulary exercise."
   - Exercises are filtered to the requested complexity before being
     sent as context; if none match, all exercises are used as a
     fallback.
   - Two-turn exchange per question: first the model presents the
     exercise without revealing the answer, then the student answers
     and the model evaluates it, reveals the answer, and explains the
     justification.
   - Uses Claude Haiku via LLMProxy with a strict system prompt that
     forbids changing difficulty or mixing exercise types unexpectedly.

OTHER FILES IN THE REPO (not .py, for context)
------------------------------------------------
- lemma_chapter.txt: vocabulary list with chapter assignments, used by
  Abacus.py.
- MAGI_verdict.json: sample/output of the jury deliberation.
- exercises_corrected.json: sample/output of the corrector stage.
- sample_sentences_graded_sample_sentences.csv: sample graded data.
- LICENSE: MIT license.

REQUIREMENTS
--------------------------------------
- Python 3
- pandas
- numpy
- networkx
- llmproxy (Tufts-University/LLMProxy — see note above)
- Standard library: json, re, os, pathlib, string, time, collections,
  xml.etree.ElementTree, concurrent.futures

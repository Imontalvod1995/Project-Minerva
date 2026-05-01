##########################
#                        #
#       LLM As Jury      #
#         (MAGI)         #
#           by           #
#   Ivan F. Montalvo D.  #
#                        #
##########################

# Library Imports

import json
from llmproxy import LLMProxy
from pathlib import Path
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep


# Class Definition

class Magi:

    """ 
    Magi is the Jury of LLMs that will provide different insight on the results of our
    main LLM. Each of the different models have a slight difference in the way that they're
    prompted, as in a way to create possible variety within the system. Magi is composed
    of the following three LLMs:

    * Melchior: As stated below, is the ChatGPT model.
    * Casper: As stated below, this is the Claude Haiku model.
    * Balthazar: As stated below, this is the Gemini 3 model.

    It also contains the following functions:
    
    * __init__: the init function for self methods.
    * rag_context_string_simple: Convert the RAG context list (from retrieve API)
    * query_context: loads the RAG and prepares it for the models.
    * deliberate: creates the logic which behind the deliberation system. 
    * juror_panel: operates the models
    """

    def __init__(self) -> None:

        """
        Initialization function, it contains all the variables for the self method.

        Input:
        1. self: methods and objects called in the __init__ function. (Wait a minute...)

        Output
        None
        """

        self.client = LLMProxy()
        self.script_exercise = """
                    Exercises must follow one of these three structures:

                    1. VOCABULARY — Complete the sentence
                            The student must fill in a blank with the correct word.
                            Example:
                                "Dionysius tum erat ________(tyrant) Syracusanorum."
                                Answer: "tyrannus"
                        Reject if: the blank is missing, multiple blanks exist, or the answer accepts
                        synonyms without justification.

                    2. GRAMMAR — Multiple choice (single option)
                        The student must select one correct option from a list to complete or fix a sentence.
                            Example:
                                "De mortuis nihil nisi bonum ________."
                                a) dixo  b) dicimur  c) dicamus  d) dictum
                                Answer: c) dicamus
                        Reject if: more than one option is correct, no single answer is clearly right,
                        or the distractors are implausible.

                    3. MAIN VERB — True or False
                        The student must identify whether the underlined word is the main verb of the sentence.
                            Example:
                                "Morti Socratis semper illacrimo, [legens] Platonem." → True or False?
                                Answer: False (the main verb is "illacrimo", not "legens")
                        Reject if: no word is underlined, the sentence is ambiguous, or both True and False
                        could be defended.

                    Any exercise that does not match one of these three structures must be tagged as
                    [off-topic] and rejected.
                    """
        self.script_verdict = """
            Return ONLY a JSON array with one object per exercise, in the exact order given.
            No preamble, no markdown fences, no wrapper keys — just the raw array.
            Each object must follow this schema:

                [
                    {
                        "exercise_id": 1,
                        "verdict": "APPROVE" | "FLAG" | "REJECT",
                        "confidence": 0–1,
                        "priority": 1–5,
                        "tags": ["..."],
                        "summary": "...",
                        "errors": [
                        { "type": "conceptual"|"factual"|"formatting", "severity": "critical"|"minor", "description": "..." }
                                ]
                    },
                    ...
                ]

                Rules:
                    - One object per exercise — never collapse multiple exercises into one
                    - "verdict": "APPROVE" if correct, "FLAG" if borderline, "REJECT" if clearly wrong
                    - "confidence": float 0–1
                    - "priority": 1–5 (1 = urgent review, 5 = low)
                    - "tags": array from [correct, incomplete, off-topic, ambiguous,
                        factual-error, conceptual-error, formatting-error, needs-rewrite]
                    - "summary": one sentence, max 15 words
                    - "errors": omit entirely for APPROVE; max 5 for REJECT, max 2 for FLAG
                """ 
        self.base_dir = Path(__file__).parent
        self.batch_size = 5


    def chunk_exercises(self, exercises:list, size:int)-> list:

        """
        This function helps to chunk the the exercises into
        a more manageable format for the jurors.

        Input:
        1. self: methods and objects called in the __init__ function.
        2. exercises: list that contains all the exercises to evaluate.
        3. size: int that sets the length of the batches.

        Output
        1. List that, through list comprehension splits a list of 
        exercises into sub-lists of the length of size.
        """
        return [exercises[i:i + size] for i in range(0, len(exercises), size)] 


    def rag_context_string_simple(self, context) -> str:

        """
        Convert the RAG context list (from retrieve API)
        into a single plain-text string that can be appended to a query.
        This function was taken from the example file retrieve_and_generate.py

        Input:
        1. self: methods and objects called in the __init__ function.
        2. context: str that is the context file in a readable form.

        Output:
        1. context_string: str that contains the context modified for juror consumption.
        """
    # If retrieve returned a plain string, wrap it and return early, else returns it empty
    
        if isinstance(context, str):
            return "\nThe following is additional context required to answer the query:\n" + context
        if not context:
            return ""
        
        context_string = ""

        i = 1

        for collection in context:
    
            if not context_string:
                context_string = """The following is additional context that is required to answer the user's query."""

            if isinstance(collection, str):
                context_string += "\n#{} {}\n".format(i, collection)

            elif isinstance(collection, dict):
                context_string += "\n#{} {}\n".format(i, collection.get('doc_summary', ''))
                j = 1

                for chunk in collection.get('chunks', []):
                    context_string += "\n#{}.{} {}\n".format(i, j, chunk)
                    j += 1
        i += 1

        return context_string

    
    def query_context(self) -> None:

        """
        This function loads the RAG and prepares it for the models.
        
        Input:
        1. self: methods and objects called in the __init__ function.

        Output:
        2. self.rag_context: str variable that contains the context.
        """
        
        address_exercises = self.base_dir/"RAG"/"exercises.json"
        address_rag = self.base_dir/"RAG"/"RAG_Score.json"        

    # Importing the exercises:

        with open(address_exercises, mode="r", encoding="utf-8") as t:
            all_exercises = json.load(t)

    # Store batches in self so the jurors can access to them without issue:

        print(f"[DEBUG] Loaded {len(all_exercises)} exercises")          # should print 30
        
        self.exercise_batches = self.chunk_exercises(all_exercises, self.batch_size)
        
        print(f"[DEBUG] Created {len(self.exercise_batches)} batches")   # e.g. 6 batches of 5
        print(f"[DEBUG] Batch sizes: {[len(b) for b in self.exercise_batches]}")
        
    # Importing RAG context:

        with open(address_rag, mode='r', encoding='utf-8') as y:
            rag_context = y.read()

        self.client.upload_text(rag_context, session_id="RAG", strategy="smart")

        print("Thinking...")
        sleep(20)

    # Single RAG retrieval for the whole batch
        rag_results = self.client.retrieve(
            query="Judge Latin exercises for correctness",
            session_id='RAG',
            rag_threshold=0.6,
            rag_k=4
            )

    # One shared query for all jurors
        self.rag_context = self.rag_context_string_simple(rag_results)


    def call_juror(self, model:str, system: str, session_id:str, batch_query:str)-> list:           
        
        """
        Function that allows for sending batch by batch to the juror.
        It returns a list of verdict dictionaries.
        Input:
        1. self: Methods and objects called in the __init__ function.
        2. model: str that defines the LLM to use.
        3. system: str that is the system prompt.
        4. session_id: str that defines the session id to use
        5. batch_query: str that contains the query with the batch of exercises.

        Output:
        1.

        """
        
        response = self.client.generate(
            model = model,
            system = system,
            query = batch_query,
            temperature = 0.5,
            lastk = 0,
            session_id = session_id,
            rag_usage = True
        )
        raws = response.get("result", "")
        match = re.search(r'\[.*\]', raws, re.DOTALL)
        
        if not match:
                raise ValueError(f"[{session_id}] No Json array found in response")
        
        return json.loads(match.group(0)) # type: ignore


    def Melchior(self) -> pd.DataFrame: 
    
        """
        As stated above, this is the ChatGPT member of Jury.
        Its personality is supposed to be more severe than the other
        two judges.

        Input:
        1. self: methods and objects called in the __init__ function.

        Output:
        1. melchior_maxim: pandas DataFrame that contains the opinion of the juror.
        """
        
        melchior_focus = """
        Focus esecially on conceptual accuracy: verify that the correct answer is linguistically sound and 
        that wrong answers are clearly incorrect.
        """
        
        rows_melchior = []
        melchior_global_ids = 1

        melchior_system = ("""
        You are a strict academic juror evaluating Latin exercises.
        Analyze each exercise critically and identify all mistakes.
        Return a JSON array with one verdict object per exercise, 
        in the same order they were given.
        """ + self.script_exercise + melchior_focus + self.script_verdict)

        for mel_batch in self.exercise_batches:
            
            mel_batch_query = self.rag_context + "\nJudge the following exercises:\n" + json.dumps(mel_batch, indent=2)
            
            try:
                verdicts_melchior = self.call_juror(
                    model = "4o-mini",
                    system = melchior_system,
                    session_id = "Magi_Iudicantes_Session_Melchior",
                    batch_query= mel_batch_query
                            )
                
                for mel_ver in verdicts_melchior:
                    rows_melchior.append({
                        "exercise_id": melchior_global_ids,
                        "verdict": mel_ver.get("verdict"),
                        "confidence": mel_ver.get("confidence"),
                        "priority": mel_ver.get("priority"),
                        "tags": mel_ver.get("tags",[]),
                        "summary": mel_ver.get("summary"),
                        "errors": mel_ver.get("errors",[])
                        })
        
                    print(f"[Melchior] Exercise #{melchior_global_ids} judged: {mel_ver.get('verdict')}")
                    melchior_global_ids += 1

            except (ValueError, json.JSONDecodeError) as er:
                print(f"[WARN] Melchior batch failed: {er}")
                melchior_global_ids += len(mel_batch)

        melchior_maxim = pd.DataFrame(rows_melchior)

        return melchior_maxim
    

    def Casper(self) -> pd.DataFrame:
    
            """
            As stated above, this is the Claude Haiku member of Jury.
            It's personality is supposed to be more benevolent than the other
            members of the Jury.

            Input:
            1. self: methods and objects called in the __init__ function

            Output:
            2. casper_maxim: pandas DataFrame that contains the opinion of the juror.
            """
            
            casper_focus = """Focus especially on sentence complexity: verify that an exercise 
            is similar in difficulty to the context provided.  
                        """

            rows_casper = []
            casper_global_ids = 1
            casper_system = ("""
                You are a benevolent academic juror evaluating a single Latin exercise.
                Analyze it benevolently and identify all mistakes.
                """ + self.script_exercise + casper_focus + self.script_verdict) 

            for cas_batch in self.exercise_batches:
                
                cas_batch_query = self.rag_context + "\nJudge the following exercises:\n" + json.dumps(cas_batch, indent=2)

                try:
                    verdicts_casper = self.call_juror(
                        model = "us.anthropic.claude-3-haiku-20240307-v1:0",
                        system = casper_system,
                        session_id = "Magi_Iudicantes_Session_Casper",
                        batch_query=cas_batch_query
                        )
    
                    for cas_ver in verdicts_casper:
                        rows_casper.append({
                        "exercise_id": casper_global_ids,
                        "verdict":     cas_ver.get("verdict"),
                        "confidence":  cas_ver.get("confidence"),
                        "priority":    cas_ver.get("priority"),
                        "tags":        cas_ver.get("tags", []),
                        "summary":     cas_ver.get("summary"),
                        "errors":      cas_ver.get("errors", [])
                            })
            
                        print(f"[Casper] Exercise #{casper_global_ids} judged: {cas_ver.get('verdict')}")
                        casper_global_ids += 1
                
                except (ValueError, json.JSONDecodeError) as err:
                    print(f"[WARN] Casper batch failed: {err}")
                    casper_global_ids += len(cas_batch)
            
            casper_maxim = pd.DataFrame(rows_casper)

            return casper_maxim
    

    def Balthazar(self) -> pd.DataFrame:
    
            """
            As stated above, this is the Gemini member of Jury. 
            It's personality is supposed to be the balanced 
            and neutral member of the Jury.

            Input:
            1. self: methods and objects called in the __init__ function

            Output:
            2. balthazar_maxim: pandas DataFrame that contains the opinion of the juror.
            """
            
            balthazar_focus = """
            Focus especially on quotation accuracy: verify that each of the exercises
            contain as part of the delivered data the textual source, such as author,
            book, line.
                            """
            
            balthazar_system = ("""
                You are a balanced academic juror evaluating a single Latin exercise.
                Analyze it neutrally and identify all mistakes.
                """ + self.script_exercise + balthazar_focus + self.script_verdict)

            rows_balthazar = []
            
            balthazar_global_ids = 1

            for bal_batch in self.exercise_batches:

                bal_batch_query = self.rag_context + "\nJudge the following exercises:\n" + json.dumps(bal_batch, indent=2)

                try:
                    verdicts_balthazar = self.call_juror(
                        model = "gemini-2.5-flash-lite",
                        system = balthazar_system,
                         session_id = "Magi_Iudicantes_Session_Balthazar",
                        batch_query=bal_batch_query
                       )
                
                    for bal_ver in verdicts_balthazar:
                        rows_balthazar.append({
                        "exercise_id": balthazar_global_ids,
                        "verdict":     bal_ver.get("verdict"),
                        "confidence":  bal_ver.get("confidence"),
                        "priority":    bal_ver.get("priority"),
                        "tags":        bal_ver.get("tags", []),
                        "summary":     bal_ver.get("summary"),
                        "errors":      bal_ver.get("errors", [])
                                })
            
                        print(f"[Balthazar] Exercise #{balthazar_global_ids} judged: {bal_ver.get('verdict')}")
                        balthazar_global_ids += 1

                except (ValueError, json.JSONDecodeError) as eo:
                    print(f"[WARN] Balthazar batch failed: {eo}")
                    balthazar_global_ids += len(bal_batch)
                
            balthazar_maxim = pd.DataFrame(rows_balthazar)

            return balthazar_maxim
    

    def deliberate(self, opinions:dict) -> pd.DataFrame:

        """
        This function creates the logic which behind the deliberation
        system. 
        
        Input: 
        
        1. Opinions: dictionary with the verdict of all three members 
        of the panel.

        Output:

        1. deliberation: pandas DataFrame with the final deliberance of each juror.  
        """
        
    # Parse each juror's response into a list of per-exercise dicts
        deliberation_rows = []

    # Get the number of exercises from the first juror's DataFrame
        num_exercises = max(len(df_opinions) for df_opinions in opinions.values())

        for i in range(num_exercises):

            votes = {"APPROVE": 0, "FLAG": 0, "REJECT": 0}
            priority_scores = []
            all_tags = []
            critical_errors = []

            for juror, df in opinions.items():

                if i >= len(df):
                    print(f"[WARN] {juror} missing exercise #{i+1}, skipping.")
                    continue

                row = df.iloc[i]
                verdict = str(row.get("verdict", "")).upper()

                if verdict in votes:
                    votes[verdict] += 1

                if row.get("priority") is not None:
                    priority_scores.append(row["priority"])

                all_tags.extend(row.get("tags", []))

                for error in row.get("errors", []):
                    if error.get("severity") == "critical":
                        critical_errors.append({
                            "juror": juror,
                            "type": error.get("type"),
                            "description": error.get("description")
                        })

            has_critical = len(critical_errors) > 0

            if has_critical:
                final = "REJECTED"
            elif votes["APPROVE"] > votes["REJECT"] + votes["FLAG"]:
                final = "APPROVED"
            elif votes["REJECT"] > 0:
                final = "REJECTED"
            else:
                final = "FLAGGED"

            deliberation_rows.append({
                "exercise_id":    i + 1,
                "final_verdict":  final,
                "votes":          votes.copy(),
                "avg_priority":   round(sum(priority_scores) / len(priority_scores), 2) if priority_scores else None,
                "tags":           list(set(all_tags)),
                "critical_errors": critical_errors,
                "total_jurors":   sum(votes.values())
            })

        deliberation_df = pd.DataFrame(deliberation_rows)

        return deliberation_df


    def juror_panel(self)-> pd.DataFrame:

        """
        This function operates the models.
        
        Input:
        1. self: methods and objects called in the __init__ function.

        Output:
        2. MAGI_verdict: dictionary containing the case, the opinion,
        and verdict by the three jurors.
        """

        self.query_context()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self.Melchior):  "Melchior",
                executor.submit(self.Casper):    "Casper",
                executor.submit(self.Balthazar): "Balthazar"
                }
            opinions = {}
            
            for future in as_completed(futures):
                juror = futures[future]
                opinions[juror] = future.result()
        
        opinions = {j: df for j, df in opinions.items() if not df.empty}
                
        if not opinions:
                print("[ERROR] All jurors returned empty DataFrames.")
                return pd.DataFrame()


            # Diagnose each juror's output before deliberating
        for juror, df in opinions.items():
            print(f"\n[DEBUG] {juror} DataFrame shape: {df.shape}")
            print(f"[DEBUG] {juror} columns: {df.columns.tolist()}")
            print(f"[DEBUG] {juror} head:\n{df.head()}")

        print("Deliberating...")
        
        MAGI_verdict = self.deliberate(opinions)

        print(f"\n[DEBUG] Deliberation DataFrame shape: {MAGI_verdict.shape}")
        print(f"[DEBUG] Deliberation columns: {MAGI_verdict.columns.tolist()}")

    # Only print summary columns if deliberation succeeded
        if not MAGI_verdict.empty:
            print(MAGI_verdict[["exercise_id", "final_verdict", "avg_priority", "total_jurors"]].to_string(index=False))
        else:
            print("[ERROR] Deliberation returned an empty DataFrame.")

        MAGI_verdict.to_json("MAGI_verdict.json", orient="records", force_ascii=False, indent=2)

        return MAGI_verdict

if __name__ == "__main__":

    MAGI = Magi()
    client = LLMProxy()

    run = MAGI.juror_panel()
    print(run)
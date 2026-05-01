########################
#        Abacus        #
#                      #
#          by          #
#                      #
#  Ivan F. Montalvo D. #
########################

# Library import

from collections import deque
import xml.etree.cElementTree as ET
import numpy as np
import networkx as nx
import os
import pandas as pd

# Class creation

class Abacus:

    """
    Abacus is the sentence complexity calculator. It doesn't run during the
    user interaction, since it creates the RAG that will be used by the generator
    and Magi. Its funtions are:

    * __init__: Initialization function.
    * safe_int: checks if the head of the graph is an int.
    * kahn_sorter: provides the sorting for the DataFrame.
    * decoder: process the XML object and extracts all the useful information. 
    * grapher: process the XML object into a graph and analyses its shape.
    * lexicon: reads the vocabulary list and provides Pandas DataFrame.
    * counter: takes all the possible patterns for verbal morphology and compares .
    them with each of those that appear in the XML object.
    * main: calls all the other functions.
    """

    def __init__(self) -> None:

        """
        Initialization function, it contains all the variables for the self method.

        Input:
        1. self: methods and objects called in the __init__ function. (Wait a minute...)

        Output
        None
        """

        self.lemma_chapter = pd.read_csv("lemma_chapter.txt")
        self.address = os.path.join(os.path.dirname(__file__), "HTML")
        self.files = [f for f in os.listdir(self.address) if f.endswith('.xml')]


    def safe_int(self, value, default=0) -> int:

        """
        This function helps to check if the head is an int.
        Else, it can return a 0.

        Input:
        1. self: methods and objects called in the __init__ function.
        2. value: A value from the graph that needs to be check. 

        Output:
        1. value: A value from the graph that needs to be check.
        2. default: default value, 0, if the number is a None or empty.
        """

        try:
            return int(value)
        
        except (TypeError, ValueError):
            return default


    def kahn_sorter(self, df, metrics) -> pd.DataFrame:
        
        """
        This function provides the sorting for the DataFrame.
        It first transform the function into a Directed Acyclic Graph 
        by vectorizing the rows. It uses this logic: 
            
            for each pair (i, j), 
            i dominates j if all(vals[i] <= vals[j]) 
            Use broadcasting: (n, 1, m) <= (1, n, m) -> (n, n, m).
        
        This step is vital, since DataFrames are not sortable through
        this method.

        After the graph is created, it is sorted using Kahn's algorithm.
        This was chosen, because the graphlib topological sorter
        was too slow. Then it is turned again as a DataFrame.
        Lastly, they're assigned complexity value given
        its topological position within the graph.

        Input:
        1. self: methods and objects called in the __init__ function.
        2. df: Pandas DataFrame with the data that is going to be converted.
        3. metrics: list of strings that allow for reading the DataFrame.
        
        Output:
        1. sorted_df: The resulting DataFrame topologically sorted by the values given.  
        """
    
    # Copy of the DataFrame so it doesn't get ruined
        df = df.copy()
        df[metrics] = df[metrics].fillna(0)
    
        vals = df[metrics].to_numpy()
        n = len(vals)

    # Creting the Graph
        print("Creating graph...")
        lte = vals[:, None, :] <= vals[None, :, :]
        dom = lte.all(axis=2)
        strict = dom & ~dom.T
        np.fill_diagonal(strict, False)
    
        in_degree = strict.sum(axis=0)
        successors = [np.where(strict[i])[0].tolist() for i in range(n)]

    # Sorting the graph
        print("Graph created, sorting...")
        queue = deque(np.where(in_degree == 0)[0])
        order = []
    
        while queue:
            node = queue.popleft()
            order.append(node)
            for successor in successors[node]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

    # resulting DataFrame using the sort as ordering mechanism
        sorted_df = df.iloc[order].reset_index(drop=True)
    
    # Assign complexity 1–10 based on topological position
        complexity = [int(np.ceil((i + 1) / n * 10)) for i in range(n)]
        sorted_df["COMPLEXITY"] = complexity
    
        return sorted_df


    def decoder(self) -> pd.DataFrame:

        """
        This function process the XML object and extracts all the useful
        information. It also appends the file name to the resulting DataFrame.

        Input:
        1. self: methods and objects called in the __init__ function.

        Output:
        1. decoder_df: pandas DataFrame that contains the text name,
        the sentence number, sentence text, the word form, and lemma.
        """
    
        rows = []

        for file in self.files:
            full_path = os.path.join(self.address, file)
            tree = ET.parse(full_path)

            file_name = file.replace('.xml', '')

        # iterate over sentences
        
            for sentence in tree.findall('.//sentence'):
            
                sentence_id = sentence.get("id")

        # reconstruct full sentence text
        
                words = [w.get('form') for w in sentence.findall('word')]
                sentence_text = " ".join(words) #type: ignore

        # iterate over words inside sentence
        
                for word in sentence.findall('word'):
                    rows.append({
                    "FILE": file_name,
                    "SENTENCE_NUMBER": sentence_id,
                    "SENTENCE_TEXT": sentence_text,
                    "word_form": word.get("form"),
                    "LEMMA": word.get("lemma"),
                    "POSTAG": word.get("postag")
                    })

        decoder_df = pd.DataFrame(rows)

        print("Data Collected! Sanity check!")
        
        return decoder_df
    

    def grapher(self)-> pd.DataFrame:

        """
        This function process the XML object into a graph and analyses its shape.

        Input:
        1. self: methods and objects called in the __init__ function.

        Output:
        1. feature_data: pandas DataFrame that contains all the shape metrics.
        The used metrics in this iteration are:  
            a. diameter_tree: int, the lenght of the sentence tree.
            b. depth_tree: int, the subordination depth of the sentence tree.
        """
        
        print("Loading grapher...")
        
        labels = [
            "FILE",         
            "SENTENCE_NUMBER",
            "SENTENCE_TEXT",
            "edges",
            "average_degree",
            "global_efficiency",
            "DIAMETER",
            "mean_eccentricity",
            "density",
            "TREE_DEPTH"
            ]

        text_features = []

    # Get sentence metadata from decoder
        decoder_df = self.decoder()
        
        print("Creating the graph...")
        
        for file in self.files:
            full_path = os.path.join(self.address, file)
            tree = ET.parse(full_path)
            file_name = file.replace('.xml', '')

            for sentence in tree.findall('.//sentence'):

                sentence_id = sentence.get("id")

    # Pull sentence text from decoder output
                match = decoder_df[
                (decoder_df["FILE"] == file_name) &
                (decoder_df["SENTENCE_NUMBER"] == sentence_id)
                ]

                if match.empty:
                    continue

                sentence_text = match["SENTENCE_TEXT"].iloc[0]

                edge_list = []
                
                for word in sentence.findall('word'):
                    
                    print("Finding a node...")
                    
                    head = self.safe_int(word.get("head"))
                    wid = self.safe_int(word.get("id"))

                    if head != 0:
                        edge_list.append((head, wid))

                if not edge_list:
                    continue

    # Undirected graph for structural metrics
                G = nx.Graph(edge_list)

                if not nx.is_connected(G):
                    largest = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest).copy()

    # Directed graph for tree depth
                DG = nx.DiGraph(G.edges())

    # Find the true root (word whose head == 0)
                root_candidates = [
                    self.safe_int(w.get("id"))
                    for w in sentence.findall('word') 
                    if self.safe_int(w.get("head")) == 0 
                    and self.safe_int(w.get("id")) in G.nodes()
                    ]

                if not root_candidates:
                    continue

                root = root_candidates[0]

                degrees = [deg for _, deg in G.degree()]
                
                features = [
                    file_name,
                    sentence_id,
                    sentence_text,
                    G.number_of_edges(),
                    np.mean(degrees),
                    nx.global_efficiency(G),
                    nx.diameter(G),
                    np.mean(list(nx.eccentricity(G).values())), #type: ignore
                    nx.density(G),
                    max(nx.single_source_shortest_path_length(DG, root).values())
                ]

                text_features.append(features)
        
        print("Graph completed!")

    # Transform the resulting data into a DataFrame    
        feature_data = pd.DataFrame(text_features, columns=labels)

        print("Graph features computed!")

        return feature_data


    def lexicon(self) -> pd.DataFrame:

        """
        This function reads the vocabulary list and provides Pandas DataFrame
        containing the text title, sentence number, sentence text, found lemma,
        and chapter. it takes the list of lemma called lemma_chapter, and the
        results from the decoder function.

        Input:
        1. self: methods and objects called in the __init__ function.

        Output:
        1. lemma_data: Pandas DataFrame, that contains the value of the words 
        that have been found to match on the lexicon as well as the sentence
        information.
        """
        
        print("Loading lexicon...")
        dfcopy = self.lemma_chapter.copy()
        decoder_df = self.decoder()

    # Creating base DataFrame
        presdf = decoder_df[["FILE", "SENTENCE_NUMBER", "SENTENCE_TEXT", "LEMMA"]].dropna()

    # Match against the vocabulary list
        lemma_data = (
            dfcopy[dfcopy["LEMMA"].isin(presdf["LEMMA"])]
            .merge(presdf, left_on="LEMMA", right_on="LEMMA", how="left")
            .sort_values("SENTENCE_NUMBER")
            )

    # Put identifying columns first
        cols = ["FILE", "SENTENCE_NUMBER", "SENTENCE_TEXT"] + [
            col for col in lemma_data.columns 
            if col not in ["FILE", "SENTENCE_NUMBER", "SENTENCE_TEXT"]]
        
        lemma_data = lemma_data[cols]

        print("Created DataFrame with all Lexical Data!")

        return lemma_data
    
    
    def counter(self) -> pd.DataFrame:

        """
        This function takes all the possible patterns for verbal morphology and compares them
        with each of those that appear in the XML object under the tag "POSTAG".
        
        Input
        1. self: methods and objects called in the __init__ function.
        
        Output
        1. verb_vals: Pandas DataFrame that contains the Verb Count, and highest grade of a verb
        as well as all the sentence information. 
        """
        
        print("Loading counter...")

        permutations = { 
            1 : ("v1spia---", "v2spia---", "v3spia---", "v1ppia---", "v2ppia---", "v3ppia---"), # present indicative active
            
            2 : ("v1siia---", "v2siia---", "v3siia---", "v1piia---", "v2piia---", "v3piia---",
            "v1spip---", "v2spip---", "v3spip---", "v1ppip---", "v2ppip---", "v3ppip---", 
            "v2spma---", "v2ppma---"), # imp ind V. a,  pres imp V. a, pres ind V. p, 
            
            3 : ("v1sria---", "v2sria---", "v3sria---", "v1pria---", "v2pria---", "v3pria---",
            "v1siip---", "v2siip---", "v3siip---", "v1piip---", "v2piip---", "v3piip---",
            "v1spsa---", "v2spsa---", "v3spsa---", "v1ppsa---", "v2ppsa---", "v3ppsa---",
            "v2sfma---", "v3sfma---", "v2pfma---", "v3pfma---","v--pna---"), # perf ind V. a, imp ind V. p, pres sub V. a, fut imp V. a, inf pres V. a 
            
            4 : ("v1slia---", "v2slia---", "v3slia---", "v1plia---", "v2plia---", "v3plia---",
            "v1spsp---", "v2spsp---", "v3spsp---", "v1ppsp---", "v2ppsp---", "v3ppsp---",
            "v1srip---", "v2srip---", "v3srip---", "v1prip---", "v2prip---", "v3prip---",
            "v1sisa---", "v2sisa---", "v3sisa---", "v1pisa---", "v2pisa---", "v3pisa---","v--pnp---"), # plup ind V.a, pres sub V. p, perf ind V. p, imp sub V. a, inf pres V. p          
            
            5 : ("v1sfia---", "v2sfia---", "v3sfia---", "v1pfia---", "v2pfia---", "v3pfia---",
            "v1slip---", "v2slip---", "v3slip---", "v1plip---", "v2plip---", "v3plip---",
            "v1sfip---", "v2sfip---", "v3sfip---", "v1pfip---", "v2pfip---", "v3pfip---",
            "v1srsa---", "v2srsa---", "v3srsa---", "v1prsa---", "v2prsa---", "v3prsa---",
            "v1sisp---", "v2sisp---", "v3sisp---", "v1pisp---", "v2pisp---", "v3pisp---",
            "v2spmp---", "v2ppmp---", "v--rna---"), # fut ind V. a, plup ind V. p, fut ind V. p, perf sub V. a, imp sub V. p, pres imp V. p, inf perf V. a
            
            6 : ("v1srsp---", "v2srsp---", "v3srsp---", "v1prsp---", "v2prsp---", "v3prsp---", 
            "v1slsa---", "v2slsa---", "v3slsa---", "v1plsa---", "v2plsa---", "v3plsa---",
            "v1stip---", "v2stip---", "v3stip---", "v1ptip---", "v2ptip---", "v3ptip---",
            "v1stia---", "v2stia---", "v3stia---", "v1ptia---", "v2ptia---", "v3ptia---",
            "v2sfmp---", "v3sfmp---", "v2pfmp---", "v3pfmp---"), # perf sub V. p, fut impe V. p, plup sub V. a, fut_per ind V. p, fut_perf ind V. a
            
            7 : ("v1slsp---", "v2slsp---", "v3slsp---", "v1plsp---", "v2plsp---", "v3plsp---")} # pluperfect subjunctive passive
        
        patterns_df = pd.DataFrame({
            "GRADE": list(permutations.keys()),
            "POSTAG": list(permutations.values())}).explode("POSTAG")
        
        
        decoder = self.decoder()

        pre_postagdf = decoder[["FILE", "SENTENCE_NUMBER", "SENTENCE_TEXT", "POSTAG"]].dropna()

        postag_data = pre_postagdf.merge(patterns_df, on="POSTAG", how="left")

    # Aggregate per sentence
        verb_vals = (
            postag_data
            .groupby(["FILE", "SENTENCE_NUMBER", "SENTENCE_TEXT"])
            .agg(
                VERB_COUNT = ("GRADE", lambda g: g.notna().sum()),
                HIGHEST_GRADE = ("GRADE", lambda g: int(g.max()) if g.notna().any() else 0),
                ).reset_index().sort_values("SENTENCE_NUMBER"))

        print("Created DataFrame with the Verbal Count!")
        
        return verb_vals


    def main(self)-> None:

        """
        This function calls all the other functions and creates the
        file that is going to be fed to the LLM as RAG.

        Input:
        1. self: methods and objects called in the __init__ function.

        Output:
        2. RAG_Score: .json file that contains the scores of each of the sentences.
        It is design to be fed as RAG to the generator.  
        """

        print("Running lexicon...")
        lexicon_df = self.lexicon()

        print("Running grapher...")
        grapher_df = self.grapher()

        print("Running counter...")
        counter_df = self.counter()


    # Find the max chapter values per sentence
        chapter_max = (
            lexicon_df.groupby(["FILE", "SENTENCE_NUMBER", "SENTENCE_TEXT"])["CHAPTER"]
            .max()
            .reset_index()
            .rename(columns = {"CHAPTER": "CHAPTER_MAX"})
            )

    # Merge on file, sentence ID and sentence text
        merged_df = chapter_max.merge(
            grapher_df[["FILE", "SENTENCE_NUMBER", "SENTENCE_TEXT", "DIAMETER", "TREE_DEPTH"]],
            on = ["FILE", "SENTENCE_NUMBER", "SENTENCE_TEXT"],
            how = "left"
            )
        
    # Merge again on file, sentence ID and sentence text, this time for highest verb and verb count.

        twice_merged_df = merged_df.merge(counter_df[["FILE", "SENTENCE_NUMBER", "SENTENCE_TEXT", "VERB_COUNT", "HIGHEST_GRADE"]],
                                          on =["FILE", "SENTENCE_NUMBER", "SENTENCE_TEXT"],
                                          how="left")

    # Adding zeroes just in case.
        
        twice_merged_df[["VERB_COUNT", "HIGHEST_GRADE"]] = (twice_merged_df[["VERB_COUNT", "HIGHEST_GRADE"]]
                                                            .fillna(0).astype(int))

    # Creating a list with the set of rows that are going to be fed to the sorter.

        metrics = ["CHAPTER_MAX", "DIAMETER", "TREE_DEPTH", "VERB_COUNT", "HIGHEST_GRADE"]

        print("Calling the graphing and sorting algorithm...")
  
    # Topological sort over a partial-order complexity graph, uses Kahn's Algorithm
  
        result_df = self.kahn_sorter(twice_merged_df, metrics)
    
    # Serialise to JSON for RAG consumption
        result_df.to_json(
            "RAG_Score.json",
            orient = "records",
            indent = 2,
            force_ascii = False)

        print("Done! RAG_Score.json written.")


if __name__ == "__main__":

 abacus = Abacus()
 run = abacus.main()


 
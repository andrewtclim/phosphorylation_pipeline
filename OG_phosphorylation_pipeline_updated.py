import requests
import json
import pandas as pd
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
import re
import numpy as np
import logging
import sys
import time
from datetime import timedelta

log_time = {"pre_processing_time": [], "num_of_seconds": [], "num_entries_processed":[],"post_processing_time":[]}
start_time = time.time()
# Retrieving company API key
with open("versa_key_copy") as file:
    apikey = file.read().strip()

apiversion = "2024-10-21"
RESOURCE_ENDPOINT='https://unified-api.ucsf.edu/general'
test_deployment = 'gpt-4.1-2025-04-14'

llm_shared = AzureChatOpenAI(
    api_key=apikey,
    api_version=apiversion,  
    azure_endpoint=RESOURCE_ENDPOINT,
    deployment_name=test_deployment,
    temperature=0.1,
    model="gpt-4.1",
)

substrate_prompt = ChatPromptTemplate.from_messages([("system","""You are an expert in extracting **kinase-substrate phosphorylation interactions** from scientific text.
Your task:
- Extract only phosphorylation-related kinases or dephosphorylated-related phosphotases.
- Make sure to focus on key phrases such as **Phosphorylated by** or **dephosphorylated by** as the kinase name will be in that sentence.
- Exclude transcription-related interactions.
- Format output as: **Kinase(kinase), Protein(Phosphotase), aminoacidlocation(location), PubMed ID**.
- If no PubMed ID is available, return "N/A".
- Exclude interactions where an interaction was enhanced by phosphorylation.
- Exclude interactions where a protein regulated phosphorylation events.
- **Only label a protein as a phosphotase if the phrase explicitly states "dephosphorylated by [protein]"**.
- **Do not assume a protein interacting with a kinase is a phosphotase unless dephosphorylation is explicitly stated.**
- Only include kinase-substrate or phosphatase-substrate interactions where phosphorylation or dephosphorylation is the key action, NOT just a prerequisite for another modification (e.g., ubiquitination, acetylation, methylation). 
- If phosphorylation is only mentioned as a dependency for another event, return "N/A".- If a kinase is explicitly named as responsible for phosphorylation (e.g., "Phosphorylated by CSNK2A1"), extract the kinase regardless of additional functional context (e.g., "release from chromatin" or "suppression of transcription").
- Exclude only cases where phosphorylation is inferred indirectly without naming a specific kinase.
- Do **not** exclude kinase extractions just because additional biological effects are described.
- Do not label a protein as a **phosphatase** unless the words dephosphorylation arespecifically in the sentence
- Before labeling a protein as a phosphatase, explicitly check that the phrase contains 'dephosphorylated by' and reject any inference based on phosphorylation regulation.
- Do NOT extract a protein unless it is explicitly stated as being phosphorylated or dephosphorylated.
- If a protein is only "regulated by phosphorylation" or "affected by phosphorylation" but is NOT explicitly phosphorylated or dephosphorylated, exclude it from the output entirely.
- Only include a protein if it is directly modified by phosphorylation (substrate) or dephosphorylation (phosphatase).
- If a protein is only mentioned in relation to phosphorylation (e.g., "regulated by phosphorylation"), return N/A instead.
- Before labeling a protein as a phosphatase, check that the phrase contains 'dephosphorylated by'. If this phrase is NOT present, DO NOT label the protein as a phosphatase.
- Proteins affected by phosphorylation, but not explicitly dephosphorylated, must be excluded (return 'N/A')."
- If "Autophosphorylated" or "Autophosphorylation" is mentioned, extract the protein as {kinase_name}(kinase), since it phosphorylates itself.
- Do not exclude autophosphorylation cases just because no other kinase is named.
- Dephosphorylated by --> Phosphotase
- Phosphorylated by --> Kinase
- **Reasoning for "N/A":** If the input text does not mention phosphorylation or dephosphorylation events, or if the interaction described does not fit into the kinase-substrate or phosphatase-substrate categories, return "N/A".
- If phosphorylation is described as **"probably" or "likely"**, extract the kinase but mark the phosphorylation as **uncertain** in reasoning.

**Examples:**

1. **Input:** "Phosphorylation at Thr-161 by CAK/CDK7 activates kinase activity (PubMed:20360007). "
   **Output:** CAK/CDK7(kinase), Thr-161(location), PubMed:20360007 
             
             
   **Reasoning:** The input describes a phosphorylation event, where CAK/CDK7 phosphorylates to activate kinase activity at a location Thr-161. 
             Since the protein was phosphorylated by CAK/CDK7 it is classified as a **kinase**.

2. **Input:** "Phosphorylation at Thr-14 and Tyr-15 by PKMYT1 prevents nuclear translocation (PubMed:7569953)."
   **Output:**  PKMYT1(kinase), Thr-14(location), Tyr-15(location), PubMed:7569953
             
   **Reasoning:** The input describes a phosphorylation event, where PKYMYT1 phosphorylates to activate kinase activity at location Thr-14 and Thr-15. 
             Since the protein was phosphorylated by CAK/CDK7 it is classified as a **kinase**.

3. **Input:** Phosphorylation at Tyr-15 by WEE1 and WEE2 inhibits the protein kinase activity and acts as a negative regulator of 
     entry into mitosis (G2 to M transition) (PubMed:20360007).
   **Output:** WEE1(kinase), WEE2(kinase), Tyr-15(location), PubMed:20360007

   **Reasoning:** The input describes a phosphorylation event, where WEE1 and WEE2 phosphorylates to activate kinase activity at location Tyr-15. 
             Since the protein was phosphorylated by  WEE1 and WEE2 it is classified as a **kinase**.

4. **Input:** Phosphorylated.
   **Output:** N/A
    
   **Reasoning:** Even though the input indicates phosphorylation, it lacks the detail of which protein(s) are responsible for this modification, 
             leading us to classify the result as **N/A** 

5. **Input:** Phosphorylated at Ser/Thr residues between Ser-68 and Thr-72 in the PEST region: required for interaction with dATP-bound RRM1 and ITPR1. 
   **Output:** Ser-68(location), Thr-72(location)
             
   **Reasoning:** The input specifies phosphorylation sites at Ser-68 and Thr-72 but does not indicate which kinase or phosphatase is responsible. Since the phosphorylation locations are provided, they are extracted without assigning a specific enzyme.
             
6. **Input:** Dephosphorylated in response to apoptotic stress conditions which causes translocation of both AHCYL1 and BCL2L10 from mitochondria-associated 
             endoplasmic reticulum membranes and promotes apoptosis (PubMed:27995898)
   **Output:** N/A

   **Reasoning:** Even though the input indicates dephosphorylation at certain residues, it lacks the detail of which protein(s) are responsible for this modification, 
             leading us to classify the result as **N/A** 

7. **Input:** Autophosphorylated and phosphorylated during M-phase of the cell cycle (PubMed:10518011, PubMed:15122335, PubMed:9988268).
   **Output:** {kinase_name}(kinase/substrate), PubMed:10518011, PubMed:15122335, PubMed:9988268
             
   **Reasoning:** The input indicates autophosphorylation, which means the protein phosphorylates itself. For autophosphorylation the protein is both the kinase and the substrate.
             
8. **Input:** Stearoylated by ZDHHC6 which inhibits TFRC-mediated activation of the JNK pathway and promotes mitochondrial fragmentation (PubMed:26214738). 
             Stearoylation does not affect iron uptake (PubMed:26214738)
   **Output:** N/A
             
   **Reasoning:** There is no mention of phosphorylation or dephosphorylation.

9. **Input:**  Phosphorylation is enhanced during cell division,at which time vimentin filaments are significantly reorganized. 
   **Output:** N/A
             
 

10: **Input:** Phosphorylated. Phosphorylated in vitro by constitutive active PKN3 
    **Output:** PKN3(kinase)
             
    **Reasoning:** The input states "Phosphorylated" without specifying the responsible kinase. However, the second part clearly mentions that phosphorylation
              was performed in vitro by constitutive active PKN3, meaning PKN3 is explicitly identified as the kinase responsible for phosphorylation. 
             Even though the specific substrate or amino acid location is not provided, the presence of a named kinase performing phosphorylation justifies extracting PKN3(kinase)
              as the output.

11, **Input:** Phosphorylated by abaofj during mitosis, resulting in its release from something happening and lebron dunking with a lob from Luka'
    **Output:** abaofj(kinase) 
    
    **Reasoning:** The input specifies that abaofj is responsible for phosphorylation, making it a kinase. Even though the description includes unrelated contextual information 
             ("resulting in its release from something happening and LeBron dunking with a lob from Luka"), the essential phosphorylation event remains valid.
              Since abaofj is explicitly named as the kinase, it is extracted as abaofj(kinase). The additional context does not affect the classification of abaofj as a kinase.

12. **Input:** Probably phosphorylated by PKC; decreases single-channel open probability
    **Output:** PKC(kinase/uncertain)
     
    **Reasoning:** The input suggests that PKC is likely responsible for phosphorylation, but the use of "probably" introduces uncertainty. 
             Since a kinase is explicitly named, PKC is extracted as a kinase, but its classification is marked as uncertain to reflect the lack of definitive confirmation. 
             No specific substrate or phosphorylation site is mentioned, so only PKC is included in the output with the uncertainty tag.

13. **Input:** Autophosphorylated and phosphorylated during M-phase of the cell cycle (PubMed:10518011, PubMed:15122335, PubMed:9988268). 
    **Output:** {kinase_name}(kinase), PubMed:10518011, PubMed:15122335, PubMed:9988268
    
    **Reasoning:** The input states that the protein is autophosphorylated and phosphorylated during M-phase of the cell cycle but does not explicitly mention which kinase 
             is responsible. In cases of autophosphorylation, the protein itself acts as both the kinase and the substrate. Since the kinase name is not explicitly provided, 
             it should be represented as {kinase_name}(kinase), while the PubMed IDs are retained to provide the source of the phosphorylation event.

14. **Input:** Phosphorylation by MAPK3/1 (ERK1/2) regulates MCRIP1 binding to CTBP(s) (PubMed:25728771)
    **Output** MAPK3/1(kinase), ERK1/2(kinase), PubMed:25728771
    
    **Reasoning:** The input states "Phosphorylation by MAPK3/1 (ERK1/2)", which explicitly indicates that MAPK3/1 (ERK1/2) is responsible for phosphorylation, meaning it should be classified as a kinase.
             Both MAPK3 and ERK1/2 are names referring to kinases in this context.MCRIP1 is only affected by phosphorylation (regulates its binding to CTBP) but is NOT explicitly phosphorylated itself. 
             Since "phosphorylated by" or "dephosphorylated by** is not used in reference to MCRIP1, it should not be labeled as a substrate or phosphotase

15. **Input:** Autophosphorylated. Phosphorylated by CDC2-CCNB1 complexes on undefined serine and threonine residues.
              The phosphorylation by CDC2-CCNB1 complexes may inhibit the catalytic activity
    **Output:** {kinase_name}(kinase), CDC2-CCNB1(kinase)
             
    **Reasoning:** The input specifically states "Autophosphorylated" which means the protein acts on itself as both a kinase and a substrate. Since the kinase name is not explicitly provided, 
             it should be represented as {kinase_name}(kinase). The input also mentions "Phosphorylated by CDC2-CCNB1" which explains the CDC2-CCNB1(kinase) output.
        

---

This approach will help the model understand how to handle cases where there are **no phosphorylation/dephosphorylation interactions** and should return "N/A" when appropriate, based on the reasoning provided.

By including **reasoning**, you ensure the model can **distinguish between relevant and irrelevant information**, ultimately improving its accuracy in handling more complex or ambiguous inputs. Would you like to proceed with further adjustments or additional examples?


Now, process the following input:

{input}
""")])
substrate_parser = StrOutputParser()
substrate_chain = substrate_prompt | llm_shared | substrate_parser

abbr_prompt = ChatPromptTemplate.from_messages([  ("system", """You are an expert in protein nomenclature. 
Your task is to return only the abbreviated name of the given protein.

**Instructions:**
- Output only the abbreviated name, without any extra text.
- Do not include explanations or full names.

**Example:**
Input: The abbreviated name for the protein myomaker is MYMK.
Output: MYMK

Now, provide the abbreviation for:

{input}
""")])
abbr_parser = StrOutputParser()
abbr_chain = abbr_prompt | llm_shared | abbr_parser

def substratelist(substrates, kinase_name, chain):
    return chain.invoke({"input": substrates, "kinase_name": kinase_name})

def abbreviated_kinases(kinase, chain):
    return chain.invoke({"input": kinase})


d = [] # list for all data to be appended to
size = 500  # Number of results per query
base_url = "https://rest.uniprot.org/uniprotkb/search"
query = "reviewed:true and organism_id:9606"  # 9606 for humans
length = 1  # Number of pages to fetch


# Create a folder named 'my_folder'
folder_name = f"phosphorylation_df_{size*length}"
os.makedirs(folder_name, exist_ok=True)

# Custom class to redirect print statements to logging
class PrintLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
    
    def write(self, message):
        message = message.strip()
        if message:
            self.logger.log(self.level, message)
    
    def flush(self):
        pass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f'./{folder_name}/phosphorylation_pipeline.log',  # Optional: logs to file
    filemode='w'            # Optional: 'a' to append
)


# Redirect print to logging
sys.stdout = PrintLogger(logging.getLogger())

# Suppress logging from noisy libraries
for noisy_logger in [
    "httpx", "httpcore", "openai", "urllib3", "requests", "http.client", "uvicorn", "uvicorn.access"
]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logging.getLogger(noisy_logger).propagate = False

# Now all print() will be logged
print("Hello, this will be logged.")
print("Another print, now captured by logging.")


params = {
    "query": query,
    "format": "json",
    "size": size
}

# Initial request (without cursor)
response = requests.get(base_url, params=params)

if response.status_code != 200:
    print(f"Error: {response.status_code}")
    print(response.text)
else:
    for i in range(length):  # Fetch up to 20 pages
        data = response.json()
        d.append(data)

        # Save each batch separately
        with open(f'./{folder_name}/uniProt_data_batch_{i}_entrylength_{size*length}.json', 'w') as f:
            json.dump(data, f, indent=4)

        # Extract 'Link' header for next page
        link_header = response.headers.get("Link")
        
        if not link_header or "cursor=" not in link_header:
            print("No more pages available.")
            break  # Stop if no 'Link' header is found

        # Extract the next page URL from the Link header
        next_url = link_header.split("<")[1].split(">")[0]

        # Fetch the next page using the extracted URL
        response = requests.get(next_url)

        if response.status_code != 200:
            print(f"Error fetching next page: {response.status_code}")
            print(response.text)
            break

print(f'this is length of d {len(d)}')
print(f'Approximate total number of entries {length*size}')
print(time.time())
elapsed_seconds = time.time() - start_time
elapsed = timedelta(seconds=elapsed_seconds)
print(f'Time elapsed = {elapsed}')

preprocessing_time = time.time() - start_time
log_time["pre_processing_time"].append(preprocessing_time)

print(f"Pre processing done {preprocessing_time} seconds")

kinase_dict = {}
protein_interaction = {}
ca ={}
la = 0
uniprot_id = {}
count = 0

for data in d:

# Iterate over all results
    for entry in data.get('results', []):
        protein_name = entry['proteinDescription']['recommendedName']['fullName']['value']

        # Search for interaction data in comments
        comments = entry.get('comments', [])
        interactions = []

        for comment in comments:
            if comment.get('commentType') == "PTM":
                for text in comment.get('texts', []):
                    value = text.get('value', '')
                    if value:
                        interactions.append(value)
        
        protein =  abbreviated_kinases(protein_name,abbr_chain)
        id = entry["primaryAccession"]
        print(f'Uniprot ID: {id}')
        uniprot_id[protein] = id
        # Print kinase and interactions
        print(f"Protein: {protein}")
        print("Interactions (Kinase-Substrate Relationships):")
        print(f'This is interactions {interactions}')
        protein_interaction[protein] = interactions
        if interactions:
            count +=1
            kinase_lst = []
            elapsed_50_entries = time.time() - start_time
            log_time["num_of_seconds"].append(elapsed_50_entries)
            log_time["num_entries_processed"].append(count)
            print(f"Processed {count} entries, time taken: {elapsed_50_entries} seconds")
            for interaction in interactions:
                time.sleep(0.2)
                print(f"{substratelist(interaction,protein,substrate_chain)}")
                elapsed_seconds = time.time() - start_time
                elapsed = timedelta(seconds=elapsed_seconds)
                print(f'Time elapsed = {elapsed}')
                kinase_lst.append(substratelist(interaction,protein,substrate_chain)) # Feed interactions and protein at hand into description
        else:
            kinase_lst = ["N/A"]
            print("N/A")
            elapsed_seconds = time.time() - start_time
            elapsed = timedelta(seconds=elapsed_seconds)
            print(f'Time elapsed = {elapsed}')
        kinase_dict[protein] = kinase_lst
        la +=1
        
        print(f'Number of Interactions: {la} --- Percentage of {(la/(length*size))*100}')
        elapsed_seconds = time.time() - start_time
        elapsed = timedelta(seconds=elapsed_seconds)
        print(f'Time elapsed = {elapsed}')

                      
    print("\n" + "-"*50 + "\n")


df = pd.DataFrame.from_dict(kinase_dict, orient='index').reset_index()
df.rename(columns={'index': 'Protein',0:'Protein Interaction'}, inplace=True) # renames df
dfinteraction = pd.DataFrame.from_dict(protein_interaction, orient='index').reset_index()
uniprot_iddf = pd.DataFrame.from_dict(uniprot_id, orient='index').reset_index()
df = df[df["Protein Interaction"].apply(lambda x: "N/A" not in x)] # removes nas
df = df[["Protein","Protein Interaction"]].reset_index(drop=True) # cleans up data frame

# Keep proteins that have at least one valid interaction (even if "N/A" is present)
df = df[df["Protein Interaction"].apply(lambda x: not (x == "N/A" or (x.count("N/A") == len(x.split(",")))))]

df = df[["Protein","Protein Interaction"]].reset_index(drop=True)
df.head()


def handle_output_in_substrate(text):
    # Extract only the part after **Output:** and before **Reasoning:** (if present)
    match = re.search(r'\*\*Output:\*\*\s*(.*?)\s*(\*\*Reasoning:\*\*|$)', text, re.DOTALL)
    if match:
        text = match.group(1).strip()  # Extract only the output part
    
    # Clean up formatting issues
    text = text.replace("\n", ", ")
    text = text.replace("\n\n", ", ")  
    text = text.replace(",,", ", ")
    
    return text


df["Protein Interaction"] = df["Protein Interaction"].apply(lambda x: x.split("**Reasoning:**")[0])
df['Protein Interaction'] = df['Protein Interaction'].apply(handle_output_in_substrate)

df["Protein Interaction"] = df["Protein Interaction"].apply(lambda x: x.split(","))

def extract_pubmed_ids_from_list(substrate_list):
    pubmed_ids = []
    # Iterate over the list and find all PubMed IDs in each item
    for text in substrate_list:
        pubmed_ids.extend(re.findall(r"PubMed:\d+", text))
    
    # Return PubMed IDs as a comma-separated string
    return ", ".join(pubmed_ids)

# Apply the function to extract PubMed IDs into a new column
df['PubMed_ids'] = df['Protein Interaction'].apply(extract_pubmed_ids_from_list)
df["PubMed_ids"] = df["PubMed_ids"].apply(lambda x: x.replace(":",","))
df["PubMed_ids"] = df["PubMed_ids"].apply(lambda x: x.split(","))


def remove_pubmed(x):
    # Create a new list, removing 'PubMed' from each entry
    return [re.sub(r"^PubMed\s*[:]*", "", str(i)) for i in x if "PubMed" not in str(i)]

# Apply the function to the PubMed column
df["PubMed_ids"] = df["PubMed_ids"].apply(remove_pubmed)

df["Protein Interaction"] = df["Protein Interaction"].apply(lambda x: ", ".join(x))

def extract_interactions(row):
    match = re.search(r'PubMed:([\d,\s]+)', row)  
    if match:
        pubmed_id = match.group(1).strip()  
        interactions = re.sub(r'PubMed:[\d,\s]+', '', row).strip()  
    else:
        pubmed_id = None
        interactions = row.strip()
    return interactions


df['Protein Interaction'] = df['Protein Interaction'].apply(lambda x: pd.Series(extract_interactions(str(x))))

df['Protein Interaction'] = df['Protein Interaction'].str.strip()


dfinteraction = dfinteraction.reset_index().rename(columns={"index": "Protein", 0: "Interaction Description"}).reset_index(drop=True)
dfinteraction = dfinteraction[["Protein","Interaction Description"]]



uniprot_iddf = uniprot_iddf.reset_index().rename(columns={"index": "Protein", 0: "protein_UniprotID"}).reset_index(drop=True)
uniprot_iddf = uniprot_iddf[["Protein","protein_UniprotID"]]
uniprot_iddf = uniprot_iddf.reset_index(drop=True)
uniprot_iddf = uniprot_iddf[["Protein","protein_UniprotID"]]

# Merges cleaned up df with dfinteractions
total_df = df.merge(dfinteraction, on="Protein", how="left")


print(total_df.columns[total_df.columns.duplicated()])

# Merges cleaned up df with uniprot protein ids
total_df = total_df.merge(uniprot_iddf, on="Protein", how="left")


def extract_full_locations_corrected(text):
    return re.findall(
        r'(Ala-\d+|Arg-\d+|Asn-\d+|Asp-\d+|Cys-\d+|Glu-\d+|Gln-\d+|His-\d+|Ile-\d+|Ser-\d+|Leu-\d+|Lys-\d+|Met-\d+|Phe-\d+|Thr-\d+|Pro-\d+|Trp-\d+|Tyr-\d+|Val-\d+|C-terminus|N-terminus'
        r'|Ala|Arg|Asn|Asp|Cys|Glu|Gln|His|Ile|Ser|Leu|Lys|Met|Phe|Thr|Pro|Trp|Tyr|Val'
        r'|Ala residues?|Arg residues?|Asn residues?|Asp residues?|Cys residues?|Glu residues?|Gln residues?|His residues?|Ile residues?|Ser residues?|Leu residues?|Lys residues?|Met residues?|Phe residues?|Thr residues?|Pro residues?|Trp residues?|Tyr residues?|Val residues?'
        r'|Alanine|Arginine|Asparagine|Aspartic acid|Cysteine|Glutamic acid|Glutamine|Histidine|Isoleucine|Serine|Leucine|Lysine|Methionine|Phenylalanine|Threonine|Proline|Tryptophan|Tyrosine|Valine'
        r'|Alanine residues?|Arginine residues?|Asparagine residues?|Aspartic acid residues?|Cysteine residues?|Glutamic acid residues?|Glutamine residues?|Histidine residues?|Isoleucine residues?|Serine residues?|Leucine residues?|Lysine residues?|Methionine residues?|Phenylalanine residues?|Threonine residues?|Proline residues?|Tryptophan residues?|Tyrosine residues?|Valine residues?)',
        text,
        re.IGNORECASE  # Ensures case-insensitive matching
    )

# Apply function to extract locations
total_df["Location"] = total_df["Protein Interaction"].apply(lambda x: ", ".join(extract_full_locations_corrected(x)))

def extract_kinases(text):
    if not isinstance(text, str):
        return ""
    
    # Match kinase names, allowing:
    # - Hyphenated names (e.g., GSK3-beta)
    # - Kinase names followed by "/uncertain" (e.g., PKC(kinase/uncertain))
    # - Standard kinase names (e.g., PKC(kinase))
    kinase_matches = re.findall(r'([\w/-]+)\s*\(kinase(?:/uncertain)?\)', text, re.IGNORECASE)
    
    return ", ".join(set(kinase_matches))  # Ensure unique values

# Function to extract phosphatases (handles NaN values)
def extract_phosphatases(text):
    if not isinstance(text, str):  # Check if text is not a string (NaN or other types)
        return ""
    return ", ".join(re.findall(r'([\w/]+)\s*\(Phosphotase\)', text, re.IGNORECASE))


total_df["Kinase"] = total_df["Protein Interaction"].apply(extract_kinases)
total_df["Phosphatase"] = total_df["Protein Interaction"].apply(extract_phosphatases)

total_df["Kinase"] = total_df["Kinase"].apply(lambda x: x.split(", ") if isinstance(x, str) else [])
total_df["Phosphatase"] = total_df["Phosphatase"].apply(lambda x: x.split(", ") if isinstance(x, str) else [])

# saves total_df to csv

total_df.to_csv(f'./{folder_name}/total_df_{la}.csv')

if os.path.exists(f'./{folder_name}/total_df_{la}.csv'):
    print("saved total_df")
    elapsed_seconds = time.time() - start_time
    elapsed = timedelta(seconds=elapsed_seconds)
    print(f'Time elapsed = {elapsed}')

total_df = total_df.explode("Kinase", ignore_index=True)
total_df = total_df.explode("Phosphatase", ignore_index=True)

proteins = total_df["Protein"].drop_duplicates().tolist()
kinases =  total_df["Kinase"].drop_duplicates().tolist()

# retrieve functions using the proteins that we filtered out through Uniprot
print("Retrieving Protein functions ...")
functions = {}
for protein in proteins:
    fun = []
    print(f"Processing: {protein}")
    elapsed_seconds = time.time() - start_time
    elapsed = timedelta(seconds=elapsed_seconds)
    print(f'Time elapsed = {elapsed}')

    
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"reviewed:true AND {protein}"  # 9606 for humans
    params = { 
        "query": query,
        "format": "json",
        "size": 20
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()

        # Save to a file (optional)
        # with open('uniProt_data.json', 'w') as f:
        #     json.dump(data, f, indent=4)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        continue  # Skip to the next protein

    # Check if there are results
    if not data.get("results"):
        print(f"No data found for {protein}")
        functions[protein] = {
            "protein_name": protein,
            "functions": "No function or entry available"
        }
        continue  # Skip to the next protein

    # Extract protein name safely
    first_result = data["results"][0]  # First result from UniProt search
    protein_name = first_result.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown Protein")

    # Extract function descriptions
    for entry in data.get('results', []):
        comments = entry.get('comments', [])
        for comment in comments:
            if comment.get('commentType') == "FUNCTION":
                for text in comment.get('texts', []):
                    fun.append(text.get('value', ''))

    functions[protein] = {
        "protein_name": protein_name,
        "functions": fun if fun else "No function descriptions available"
    }

    print(f"Extracted: {protein_name}")
    print(f"Functions: {functions[protein]['functions']}")

# Retrieve Uniprot ID for kinases from the filtered Uniprot Pipeline
print("Retrieving Kinase Uniprot IDs ...")
kinases_id = {}
for kin in kinases:
    fun = []
    print(f"Processing: {kin}")
    
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"reviewed:true AND {kin}"  # 9606 for humans
    params = { 
        "query": query,
        "format": "json",
        "size": 20
    }

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()

    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        continue  # Skip to the next protein

    # Check if there are results
    if not data.get("results"):
        print(f"No data found for {kin}")
        kinases_id[kin] = {
            "kinase_name": kin,
            "kinase_UniprotID": "No kinase_id or entry available"
        }
        continue  # Skip to the next protein

    # Extract protein name safely
    first_result = data["results"][0]  # First result from UniProt search
    kin_id = first_result["primaryAccession"]
    print(f' kinase: {kin}, kinase_id: {kin_id}')

    kinases_id[kin] = {
        "protein_name": kin,
        "id": kin_id if kin_id else "No kinase_id descriptions available"
    }
    print(kinases_id)
    print(f"Extracted: {kin}")
    print(f"Functions: {kinases_id[kin]['id']}")
    elapsed_seconds = time.time() - start_time
    elapsed = timedelta(seconds=elapsed_seconds)
    print(f'Time elapsed = {elapsed}')
    
# Make a new dataframe from the functions
df = pd.DataFrame.from_dict(functions, orient="index")


# Make a new dataframe from the kinases
dfkin = pd.DataFrame.from_dict(kinases_id, orient="index")


# Filtering functions in order to get specific information out from it
keywords = ["phosphorylation","phosphorylates","phosphorylated", "phosphorylate", "kinase", "substrate","Phosphorylation",
            "Phosphorylates","Phosphorylated", "Autophosphorylation", "autophosphorylation", "Dephosphorylate"]

# Function to extract only relevant function descriptions
def extract_relevant_functions(function_list):
    if not isinstance(function_list, list):  # Ensure it's a list
        return []
    return [sentence for sentence in function_list if any(keyword in sentence.lower() for keyword in keywords)]

# Apply the function to extract relevant function descriptions
df["filtered_functions"] = df["functions"].apply(extract_relevant_functions)

df = df[["protein_name","filtered_functions"]]

df_reset = df.reset_index().rename(columns={'index': 'protein_id'})
# Merge total_df with protein 
final_df = total_df.merge(df_reset, left_on= "Protein", right_on = "protein_id", how = "inner")

#clean uo dataframe
final_df = final_df.merge(dfkin, left_on="Kinase", right_on="protein_name", how="left")
final_df = final_df.rename(columns={"id": "kinase_UniprotID","protein_name_x":"protein_full_name"})
final_df = final_df.drop(columns=["protein_name_y"])



#save finaldf as csv to folder
final_df.to_csv(f"./{folder_name}/final_df_{la}.csv")
if os.path.exists(f"./{folder_name}/final_df_{la}.csv"):
    print("saved final_df")
    elapsed_seconds = time.time() - start_time
    elapsed = timedelta(seconds=elapsed_seconds)
    print(f'Time elapsed = {elapsed}')



#Make sure you have Kinase Substrate Dataset in the right file path 
phosphosite_df = pd.read_csv('Kinase_Substrate_Dataset', sep='\t', encoding='ISO-8859-1')
pdf = phosphosite_df[(phosphosite_df["KIN_ORGANISM"] == "human") & (phosphosite_df["SUB_ORGANISM"]=="human")]
pdf_refined = pdf[["GENE","KINASE","KIN_ACC_ID","SUB_ACC_ID","SUB_GENE","SUBSTRATE","SITE_+/-7_AA"]]


duplicates = final_df.columns[final_df.columns.duplicated()].tolist()
print("Duplicate columns:", duplicates)

# making sure there are no duplicated columns
final_df = final_df.loc[:, ~final_df.columns.duplicated(keep='first')]


# merge phosphosite with uniprotdf 
f = final_df.merge(pdf_refined,left_on=["protein_UniprotID","kinase_UniprotID"],right_on=["SUB_ACC_ID","KIN_ACC_ID"])
f = f.drop_duplicates(subset=["protein_UniprotID", "kinase_UniprotID"])

f.to_csv(f"./{folder_name}/merged_matches_phsophosite_{la}.csv")

if os.path.exists(f"./{folder_name}/merged_matches_phsophosite_{la}.csv"):
    print("save merged")
    elapsed_seconds = time.time() - start_time
    elapsed = timedelta(seconds=elapsed_seconds)
    print(f'Time elapsed = {elapsed}')

pairs_to_exclude = set(tuple(x) for x in f[["Protein", "Kinase"]].values)
excluded_df = final_df[~final_df[["Protein", "Kinase"]].apply(tuple, axis=1).isin(pairs_to_exclude)] # made an exluded data frame for potential manual curation

excluded_df.to_csv(f"./{folder_name}/excluded_{la}.csv") # saved exluded df to csv

if os.path.exists(f"./{folder_name}/excluded_{la}.csv"):
    print("save excluded df")
    elapsed_seconds = time.time() - start_time
    elapsed = timedelta(seconds=elapsed_seconds)
    print(f'Time elapsed = {elapsed}')

pairs_to_include = set(tuple(x) for x in final_df[["Protein", "Kinase"]].values)



final_df["Matches Phosphosite+"] = final_df[["Protein", "Kinase"]].apply(
    lambda row: "Yes" if tuple(row) in pairs_to_exclude else "No",
    axis=1
)

adjusted_final = final_df


adjusted_final.to_csv(f"./{folder_name}/adjusted_final{la}.csv")

if os.path.exists(f"./{folder_name}/adjusted_final{la}.csv"):
    print("saved adjusted_final")
    elapsed_seconds = time.time() - start_time
    elapsed = timedelta(seconds=elapsed_seconds)
    print(f'Time elapsed = {elapsed}')


adjusted_final_merged = adjusted_final.merge(pdf_refined,left_on=["protein_UniprotID","kinase_UniprotID"],right_on=["SUB_ACC_ID","KIN_ACC_ID"],how = "left")
adjusted_final_merged

adjusted_final_merged.to_csv(f"./{folder_name}/adjusted_final_merged_{la}.csv")

if os.path.exists(f"./{folder_name}/adjusted_final_merged_{la}.csv"):
    print("saved adjusted_final_mergedf")

elapsed_seconds = time.time() - start_time
elapsed = timedelta(seconds=elapsed_seconds)
print(f'FINAL ELAPSED TIME: {elapsed}, JOB IS FINISHED...')


log_time["post_processing_time"].append(time.time() - start_time)
print(f"Post processing done {log_time['post_processing_time'][-1]} seconds")


log_time_df = pd.DataFrame(log_time)
log_time_df.to_csv(f"./{folder_name}/log_time_{la}.csv")


# System Prompt while generating the reference list for the non-factoid question answering task.
DEFAULT_SYS_PROMPT = "You are a helpful assistant."

# refer to https://github.com/babyyang525/LINKAGE-Listwise-NFQA-Evaluation/blob/main/prompt/LINKAGE.txt
LINKAGE_PROMPT_TEMPLATE = """Please impartially rank the given candidate answer to a non-factoid question accurately within the reference answer list, which are ranked in descending order of quality. The top answers are of the highest quality, while those at the bottom may be poor or unrelated.
Determine the ranking of the given candidate answer within the provided reference answer list. For instance, if it outperforms all references, output [[1]]. If it's deemed inferior to all {#num_references_en} references, output [[{#num_references_int}]].
Your response must strictly following this format: \"[[2]]\" if candidate answer could rank 2nd.
Below are the user's question, reference answer list, and the candidate answer.
Question:{#question}
Reference answer list:{#ground}
Candidate answer:{#candidate}
"""

# example of num_references_en: "four"
# example of num_references_int: 4

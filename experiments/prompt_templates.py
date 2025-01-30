#################### Prompts for LLM method ####################
LLM_SYS_PROMPT = "You are an assistant for answering questions."
LLM_PROMPT_TEMPLATE = (
    "Answer the following question.\n\n"
    "### Question\n"
    "{query}\n\n"
    "### Answer\n"
)
LLM_PROMPT_TEMPLATE_SUFFIX = "### Answer\n"

#################### Prompts for RAG method ####################
RAG_SYS_PROMPT = "You are an assistant for answering questions."
RAG_PROMPT_TEMPLATE = (
    "You are given the extracted parts of a long document and a question. "
    "Refer to the references below and answer the following question.\n\n"
    "### References\n"
    "{references}\n\n"
    "### Question\n"
    "{query}\n\n"
    "### Answer\n"
)
RAG_PROMPT_TEMPLATE_SUFFIX = "### Answer\n"

#################### Prompts for Typed-RAG ####################
# SYSTEM PROMPTS
DEFAULT_SYS_PROMPT = "You are a helpful assistant."
QUERY_ANALYST_SYS_PROMPT = "You are a query analysis assistant. Based on the query type, apply the relevant prompt to transform the query to better align with the user's intent, ensuring clarity and precision."
DEBATE_MEDIATOR_SYS_PROMPT = "You are acting as the mediator in a debate."
AGGREGATOR_SYS_PROMPT = (
    "You are an assistant tasked with aggregating answers to a question.\n"
    "You are provided with the original question and multiple question-answer pairs. "
    "These queries preserve the core intent of the original question but use varied language and structure. "
    "Your goal is to review the question-answer pairs and synthesize a concise and accurate response to the original question based on the information provided."
)

# PROMPTS FOR DEBATE-TYPE QUESTIONS
DEBATE_SUBQUERY_GENERATION_PROMPT = (
    "The input question is a debate-type question (i.e., invites multiple perspectives). "
    'As a \"Query Analyst\", please evaluate this question and proceed with the following steps.\n'
    "\n"
    "1. Extract the debate topic.\n"
    "2. Identify 2 to 5 key perspectives on this topic.\n"
    "3. Generate a sub-query reflecting each perspective’s bias.\n"
    "\n"
    "Ensure each sub-query fits a Retrieval-Augmented Generation (RAG) framework, seeking passages that align with the viewpoint.\n"
    "\n"
    "### Output format\n"
    '{{"debate_topic": {{topic}},"dist_opinion": [list of perspectives],"sub-queries": {{"opinion1": "biased sub-query for opinion1","opinion2": "biased sub-query for opinion2",...}}}}\n'
    "\n"
    "### Example\n"
    'Query: "Is Trump a good president?"\n'
    "Answer: \n"
    '{{\n'
    '   "debate_topic": "Donald Trump\'s presidency",\n'
    '   "dist_opinion": ["positive", "negative", "neutral"],\n'
    '   "sub-queries": {{\n'
    '      "positive": "Was Donald Trump one of the best presidents for economic growth?",\n'
    '      "negative": "Did Trump\'s presidency harm the U.S. economy and leadership?",\n'
    '      "neutral": "Can we assess Trump\'s tenure\'s strengths and weaknesses?"\n'
    "   }}\n"
    "}}\n"
    "\n"
    "### Input\n"
    "Query: {query}\n"
    "### Output\n"
    "Answer: \n"
)
DEBATE_MEDIATOR_PROMPT = (
    "Below is a topic and responses provided by n participants, each with their own perspective. "
    "Your task is to synthesize these responses by considering both the debate topic and each participant's viewpoint, providing a fair and balanced summary. "
    "Ensure the response maintains balance, captures key points, and distinguishes any opposing opinions. "
    'Present the answer *short and concise*, phrased in a direct format without using phrases like "participants in the debate" or "in the debate."\n'
    "\n"
    "### Input format\n"
    "- Debate topic: {{debate_topic}}\n"
    "- Participant's responses:\n"
    '  - Response 1: "{{response content}}" (Perspective: {{perspective 1}})\n'
    '  - Response 2: "{{response content}}" (Perspective: {{perspective 2}})\n'
    '  - ...\n'
    '  - Response N: "{{response content}}" (Perspective: {{perspective N}})\n'
    "\n"
    "### Output format\n"
    "A short and concise summary from the mediator’s perspective based on the discussion, phrased as a direct answer without reference to the debate structure or participants\n"
    "\n"
    "### Inputs\n"
    "Debate topic: {debate_topic}\n"
    "Participant's responses:\n"
    "{responses}\n"
    "### Output\n"
    "Summary: "
)

# PROMPT FOR EXPERIENCE-TYPE QUESTIONS
EXPERIENCE_KEYWORD_EXTRACTION_PROMPT = (
    "The input question is an experience-type question (i.e., get advice or recommendations on a particular topic.). "
    "As a \"Query Analyst\", please evaluate this question and proceed with the following steps.\n"
    "\n"
    "1. Identify the topic intended to be gathered from experience-based questions.\n"
    "2. Extract the key entities in the question, considering the intent of asking about experience, to facilitate an accurate response.\n"
    "\n"
    "### Output format\n"
    "`[\"Keyword 1\", ..., \"Keyword N\"]` (List of string, separated with comma)\n"
    "\n"
    "### Example\n"
    "Question (Input): \"What are some of the best Portuguese wines?\"\n"
    "Answer (Output): [\"Portuguese wines\", \"best\"]\n"
    "\n"
    "### Input\n"
    "Question: {query}\n"
    "### Output\n"
    "Answer: "
)

# PROMPTS FOR COMPARISON-TYPE QUESTIONS
COMPARISON_KEYWORD_EXTRACTION_PROMPT = (
    "Determine if the input query is a compare-type question (i.e., compare/contrast two or more things, understand their differences/similarities.) as a \"Query Analyst\". "
    "If so, perform the following:\n"
    "\n"
    "1. Identify the type of comparison: \"differences\", \"similarities\", or \"superiority\".\n"
    "2. Extract the subjects of comparison and represent them as specific, contextualized phrases.\n"
    "### Output format\n"
    "{{\"is_compare\": true/false, \"compare_type\": \"\", \"keywords_list\":[]}}\n"
    "\n"
    "### Example\n"
    "Query: \"Who is more intelligent than humans on earth?\"\n"
    "Analysis:\n"
    "{{\"is_compare\": true, \"compare_type\":\"superiority\", \"keywords_list\": [\"human intelligence\", \"the intelligence of other beings\"]}}\n"
    "\n"
    "### Input\n"
    "Query: {query}\n"
    "### Output\n"
    "Analysis:\n"
)
COMPARISON_GENERATOR_PROMPT = (
    "You are given the extracted parts of a long document and a question. "
    "Refer to the references below and answer the following question.\n\n"
    "The question is a compare-type with a specific comparison type and keywords indicating the items to compare.\n"
    "Answer based on this comparison type and the target keywords provided.\n"
    "\n"
    "### Inputs\n"
    "Question: {query}\n"
    "Comparison Type: {comparison_type}\n"
    "Keywords: {keywords_list_text}\n"
    "References:\n"
    "{references}\n"
    "### Output\n"
    "Answer: "
)

# PROMPT FOR REASON-TYPE QUESTIONS
REASON_QUERY_DECOMPOSITION_PROMPT = (
    "The input query is a reason-type question (i.e., a question posed to understand the reason behind a particular concept or phenomenon). "
    "As a \"Query Analyst\", please evaluate this query and proceed with the following steps.\n"
    "\n"
    "1. Break down the original instruction into multiple sub-queries that preserve the core intent but use varied language and structure. These multiple sub-queries should aim to capture different linguistic expressions of the original instruction while still aligning with its intended meaning.\n"
    "2. Create at least 2 to 5 distinct multiple sub-queries.\n"
    "\n"
    "### Output format\n"
    "`[\"sub-query 1\", ..., \"sub-query N\"]` (List of string, separated with comma)\n"
    "\n"
    "### Input\n"
    "Query: {query}\n"
    "### Output\n"
    "Multiple sub-queries: \n"
)

# PROMPT FOR INSTRUCTION-TYPE QUESTIONS
INSTRUCTION_QUERY_DECOMPOSITION_PROMPT = (
    "The input query is an instruction-type question (i.e., Instructions/guidelines provided in a step-by-step manner). "
    "As a \"Query Analyst\", please evaluate this query and proceed with the following steps.\n"
    "\n"
    "1. Break down the original instruction into multiple sub-queries that preserve the core intent but use varied language and structure. "
    "These multiple sub-queries should aim to capture different linguistic expressions of the original instruction while still aligning with its intended meaning.\n"
    "2. Create at least 2 to 5 distinct multiple sub-queries.\n"
    "\n"
    "### Output format\n"
    "`[\"sub-query 1\", ..., \"sub-query N\"]` (List of string, separated with comma)\n"
    "\n"
    "### Input\n"
    "Query: {query}\n"
    "### Output\n"
    "Multiple sub-queries: \n"
)

# AGGREGRATION PROMPT FOR REASON-TYPE AND INSTRUCTION-TYPE QUESTIONS
AGGREGATION_PROMPT = (
    "Using the information from the question-answer pairs, generate a brief and clear answer to the original question.\n"
    "\n"
    "### Inputs\n"
    "Original Question: {original_question}\n"
    "Question-Answer Pairs: \n"
    "{qa_pairs_text}\n"
    "### Output\n"
    "Aggregated Answer: "
)
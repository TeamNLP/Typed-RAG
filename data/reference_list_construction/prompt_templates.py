# System Prompt while generating the reference list for the non-factoid question answering task.
default_system_prompt = "You are a helpful assistant."

# Instruction for generating the highest standard reference answer (From Figure 8 in the LINKAGE paper)
generating_highest_reference = (
    'Given a non-factoid question:"{question}" and its answer:"{ground0}"'
    "\n"
    "Use your internal knowledge to rewrite this answer."
)

# Instruction for generating other reference answers in R sorted by quality descendingly. (From Figure 9 in the LINKAGE paper)
generating_other_references = (
    "Generate three different answers to a non-factoid question from good to bad in quality, each inferior to the golden answer I give you. "
    "Ensure that the quality gap from good to bad is very significant among these three answers. "
    "Golden answer is the reasonable and convincing answer to the question. "
    "Answer 1 can be an answer to the question, however, it is not sufficiently convincing. "
    "Answer 2 does not answer the question or if it does, it provides an unreasonable answer. "
    "Answer 3 is completely out of context or does not make any sense."
    "\n"
    "\n"
    "Here are 3 examples for your reference."
    "\n"
    "1.Non-factoid Question: how can we get concentration on something?"
    "\n"
    "Golden Answer: To improve concentration, set clear goals, create a distraction-free environment, use time management techniques like the Pomodoro Technique, practice mindfulness, take regular breaks, stay organized, limit multitasking, practice deep work, maintain physical health, and seek help if needed."
    "\n"
    "Output:"
    "\n"
    "Answer 1: Improve focus: set goals, quiet space, Pomodoro Technique, mindfulness, breaks, organization, limit multitasking, deep work, health, seek help if needed."
    "\n"
    "Answer 2: Just like and enjoy the work you do, concentration will come automatically."
    "\n"
    "Answer 3: If you are student, you should concentrate on studies and don't ask childish questions."
    "\n"
    "\n"
    "2.Non-factoid Question: Why doesn't the water fall off earth if it's round?"
    "\n"
    "Golden Answer: Earth's gravity pulls everything toward its center, including water. Even though Earth is round, gravity keeps water and everything else anchored to its surface. Gravity's force is strong enough to counteract the Earth's curvature, preventing water from falling off."
    "\n"
    "Output:"
    "\n"
    "Answer 1: This goes along with the question of why don't we fall off the earth if it is round. The answer is because gravity is holding us (and the water) down."
    "\n"
    "Answer 2: Same reason the people don't."
    "\n"
    "Answer 3: When rain drops fall through the atmosphere CO2 becomes dissolved in the water. CO2 is a normal component of the Earth's atmosphere, thus the rain is considered naturally acidic."
    "\n"
    "\n"
    "3.Non-factoid Question: How do I determine the charge of the iron in FeCl3?"
    "\n"
    "Golden Answer: Since chloride ions (Cl-) each carry a charge of -1, and there are three chloride ions in FeCl3, the total negative charge from chloride ions is -3. To balance this, the iron ion (Fe) must have a charge of +3 to ensure the compound has a neutral overall charge. Therefore, the charge of the iron ion in FeCl3 is +3."
    "\n"
    "Output:"
    "\n"
    "Answer 1: Charge of Fe in Fecl3 is 3. Iron has either 2 as valancy or 3. in this case it bonds with three chlorine molecules. therefore its valency and charge is three."
    "\n"
    "Answer 2: If two particles (or ions, or whatever) have opposite charge, then one has positive charge and one has negative charge."
    "\n"
    "Answer 3: take a piece of iron. Wrap a copper wire around the iron in tight close coils. run a charge through the wire."
    "\n"
    "\n"
    "Below are the non-factoid question, and the golden answer."
    "\n"
    "Non-factoid Question: {question}"
    "\n"
    "Golden Answer: {ground0}"
    "\n"
    "Output:"
    "\n"
)

# TEST
if __name__ == "__main__":
    QUESTION = "What is the capital of KOREA?"
    ANSWER = "Seoul"
    print("===== Test for the prompt.py =====")
    print("Question:", QUESTION)
    print("Answer:", ANSWER)

    print(
        "\n===== Instruction for generating the highest standard reference answer (From Figure 8 in the LINKAGE paper) ====="
    )
    print(generating_highest_reference.format(question=QUESTION, ground0=ANSWER))

    print(
        "\n===== Instruction for generating other reference answers in R sorted by quality descendingly. (From Figure 9 in the LINKAGE paper) ====="
    )
    print(generating_other_references.format(question=QUESTION, ground0=ANSWER))

{
    "debate_topic": "",
    "base_answer": "",
    "debate_answer": "",
    "player_meta_prompt": "You are the primary decision-maker responsible for answering questions accurately and comprehensively. In each round, you will receive feedback from a consultant, and your job will be to refine your answer based on that feedback. Each iteration should improve the quality of the answer, ensuring both clarity and correctness.\nThe question is:\n##debate_topic##",
    "moderator_meta_prompt": "You are a consultant tasked with reviewing answers provided by the primary decision-maker and provide constructive feedback to improve the answer's completeness, accuracy, and clarity. Look for any missing details, unclear explanations, or aspects that could be improved, and suggest modifications to refine the answer. Provide your feedback in a concise and actionable manner. \n The consultancy topic is: \n##debate_topic##",
    "affirmative_prompt": "##debate_topic##",
    "moderator_prompt": "Now the ##round## round of consultancy begins.\n\nThe assistant's response is: ##aff_ans## \n\nYou, as the consultant, will evaluate the answer for missing details, unclear points, or improvements. Provide concise, actionable feedback with suggested modifications. \n Now please output your response in the json format as follows: {\"Do you find the final answer provided by the assistant accurate and satisfactory?\": \"Yes or No\", \"Reason\": \"\", \"Feedback to Assistant for modifications\": \"\", \"final_answer\": \"\"}. \n Please strictly output in JSON format, do not output irrelevant content.",
    "debate_prompt": "Now improve your answer based on the consultant's feedback as follow: ##oppo_ans##\n\n"
}

from local_llm import generate_answer

context = "Six Sigma is a data-driven methodology used to improve processes."
question = "What is Six Sigma?"

answer = generate_answer(context, question)
print(answer)
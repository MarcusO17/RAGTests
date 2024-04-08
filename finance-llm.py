import ollama

stream = ollama.chat(model='finance-chat',
                     messages=[{'role':'user','content':'Give me the basics of finance'}],
                     stream = True,)

for chunk in stream:
    print(chunk['message']['content'],end='',flush=True)
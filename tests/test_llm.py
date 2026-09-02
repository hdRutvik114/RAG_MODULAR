from app.generation.llm_service import LLMservice





def llmservice():
    llmservice=LLMservice()
    
    prompt="Hi bro .do u know sdm ujire college "
    response=llmservice.generate(prompt)
    print(response)
    
llmservice()
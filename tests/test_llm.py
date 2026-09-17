from app.generation.llm_service import LLMservice


def test_llm_service():
    service = LLMservice()
    prompt = "Hi, reply with one word: 'connected'."
    response = service.generate(prompt)
    print("LLM Response:", response)
    assert response is not None
    assert len(response) > 0


if __name__ == "__main__":
    test_llm_service()
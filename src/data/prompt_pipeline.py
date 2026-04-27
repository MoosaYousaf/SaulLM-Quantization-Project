import os

def format_legal_prompt(filepath: str) -> str:
    """
    Reads a legal document and formats it for SaulLM-7B (Mistral architecture).
    Wraps the instruction and document in the required <s>[INST] ... [/INST] tags.
    """
    # Ensure file exists before attempting to open it
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: Could not locate the document at '{filepath}'. Check your relative path.")
        
    # Read raw text from mock document
    with open(filepath, "r", encoding="utf-8") as file:
        document_text = file.read().strip()
        
    # Define specific task for AI to perform on the document
    instruction = (
        "You are an expert legal AI assistant. Review the following Mutual Non-Disclosure Agreement and return "
        "exactly 3 bullet points in this order and with these headings:\n"
        "- Confidential Information:\n"
        "- Receiving Party Obligations:\n"
        "- Governing Law:\n"
        "Each bullet must be one sentence, under 25 words, and explicitly mention the key clause details."
    )
    
    # Construct Mistral-compliant prompt structure
    formatted_prompt = f"<s>[INST] {instruction}\n\n{document_text} [/INST]"
    
    return formatted_prompt

# ---------------------------------------------------------
# Local Testing Block
# ---------------------------------------------------------
# This block only runs if you execute this specific file directly.
# It will not execute when imported into your Jupyter notebook.
if __name__ == "__main__":
    # Assuming execution from the root repository directory
    test_filepath = "src/data/raw_documents/mock_nda.txt"
    
    try:
        final_prompt = format_legal_prompt(test_filepath)
        print("✅ Prompt successfully formatted for SaulLM-7B:\n")
        print("=" * 60)
        print(final_prompt)
        print("=" * 60)
    except Exception as e:
        print(f"❌ {e}")

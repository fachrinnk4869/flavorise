from rag import embed_rag
from algorithm import matching_algorithm
from gui import gui_output

def main():
    # 1. RAG
    list_rag = embed_rag()

    
    # 2. Matching algoritma
    final_result = matching_algorithm(list_rag)

    # output to UI
    gui_output(final_result) 
    

if __name__ == "__main__":
    main()


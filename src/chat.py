import sys
from pathlib import Path

# Garantir import de search.py
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from search import search_prompt


def main():
    try:
        chain = search_prompt()
    except Exception as e:
        print(f"Erro ao inicializar o serviço de busca: {e}")
        return

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta:")
    while True:
        try:
            pergunta = input("\nPERGUNTA: ").strip()
            if not pergunta:
                continue
            if pergunta.lower() in ["sair", "exit", "quit"]:
                print("Encerrando o chat.")
                break

            resposta = chain.invoke({"pergunta": pergunta})
            print(f"RESPOSTA: {resposta}")
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o chat.")
            break
        except Exception as e:
            print(f"Erro ao processar pergunta: {e}")


if __name__ == "__main__":
    main()

# Modelo Transformer Preditor de Labirintos

O objetivo deste projeto é representar labirintos como sequências de caracteres de texto e treinar um modelo de inteligência artificial com uma arquitetura de transformer capaz de compreender a topologia estrutural e espacial do mapa. A partir de um ponto inicial (`S`), o modelo deve ser capaz de ler os arredores e fornecer uma sequência de instruções direcionais (Cima, Baixo, Esquerda, Direita) que percorre os caminhos livres até alcançar uma posição de destino (`E`), garantindo que nenhuma parede (`#`) seja violada no processo. 

## Como Usar
Para instalar as dependências necessárias e preparar o ambiente de forma automática, instale o pacote root no modo iterativo:

```bash
pip install -e .
```

Para executar o script de treinamento, utilize o comando abaixo e insira as especificações pelo terminal:
```bash
python trainer.py
```

Para executar o script de avaliação, utilize o comando abaixo e insira as especificações pelo terminal:
```bash
python evaluator.py
```

## O Dataset
O dataset usa geração de dados utilizando os algoritmos Busca em Profundidade (DFS), Busca em Profundidade com Percolação e o Algoritmo de Wilson. O formato serializado inclui marcações especiais (`<LABYRINTH_START>`, `<SOLUTION_START>`, `<LABYRINTH_END>`, `<SOLUTION_END>`) para melhorar a estabilidade do treino.

## Arquitetura e Modelos
O núcleo do projeto explora e compara diferentes arquiteturas baseadas em Transformers:

*   **Decoder-Only:** Arquitetura baseada no GPT-2. Consiste em blocos com Masked Self-Attention, seguidos por uma rede Feed Forward (expansão 4x com ativação GELU), Layer Normalization e regularização via Dropout (10%). Esta arquitetura exige a predição autorregressiva direta e demonstrou a melhor eficiência e convergência.
*   **Encoder-Only:** Analisa a grid de forma bidirecional utilizando Unmasked Self-Attention para tentar resolver o mapa de uma única vez.
*   **Encoder-Decoder:** O Encoder lê a estrutura estática do labirinto obtendo um contexto global bidirecional. O Decoder usa Cross-Attention para consultar esse mapa enquanto gera passo a passo a solução.

## O Treinamento
Os modelos foram construídos em PyTorch e treinados iterativamente utilizando as seguintes estratégias:

*   **Função de Perda:** Utilizamos `CrossEntropyLoss` calculando a divergência entre os tokens. Uma otimização crucial foi a utilização do `ignore_index` para evitar que o token de preenchimento (`<PAD>`) adicionasse ruído aos gradientes, acelerando o treinamento e economizando recursos computacionais.
*   **Estratégia:** Treinamento supervisionado via *Teacher Forcing*. Ao usar o *ground truth* para alimentar as predições futuras durante a fase de treino, o modelo ganha estabilidade em sua descida de gradiente.
*   **Otimizador:** AdamW. Acoplado com o método de *Gradient Clipping* com `max_norm=1.0`, garantimos que os pesos do transformer não sofram com explosão de gradientes.
*   **Gestão e Hiperparâmetros:** Os datasets contam com 100.000 labirintos de treino e 1.000 de teste. As taxas de aprendizado foram constantes, com salvamentos por época e um sistema de *checkpointing* automático (`training_logs.json`) para retomar o treinamento já salvo a partir da última época treinada.

## Resultados
A arquitetura `Decoder-Only` obteve o maior sucesso isolado no processamento e solução do desafio.

O nosso melhor modelo atingiu uma marca de **99.6% de acurácia na solução** dos labirintos de validação. Dentro destas respostas corretas, notáveis **94.9% consistiram da rota ótima de menor caminho**. Esses resultados indicam forte evidência de que um transformer pode de fato mapear a espacialidade bidimensional a partir de um input unilateral. Todos os pesos de modelos e logs de treino são gerados de forma automática na pasta raiz `./runs/` separada pelo nome das configurações.
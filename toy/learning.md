# Pipeline de Entrada

O pipeline de entrada é o processo de tomar o texto que será dado na entrada e transformá-lo em tokens já interpretados com um significado vetorial. Ele consiste de três etapas: O encoding para fazer a tokenização, criação de um data loader para carregar os tokens em batches e o embedding para mapear os tokens em vetores.

- $Encoding \rightarrow DataSampling \rightarrow Embedding$

### Encoding

- A tokenização é o processo de transformar nossas palavras em tokens, que conseguem ser lidos e operados pela LLM, pois elas não sabem lidar diretamente com o texto.

- Para tokenizar o input, a ideia inicial é achar um encoding $X$ -> $y$ que mapeia todas as palavras $X_i$ em inteiros $y_i$.

- Um dos métodos de encoding é o **byte pair encoding**, que começa associando tokens a todos os caracteres individuais, depois os agrupa em subpalavras menores se for uma combinação frequente e o faz sucessivamente combinando subpalavras que são frequentes juntas. Palavras desconhecidas podem ser deconstruídas em tokens menores em qualquer caso.

- O processo de encoding é feito a partir de um dicionário inicial, que é o que chamamos de vocabulário. Ele possui uma amostra do tipo de texto e palavras que vamos ler.

### Data Sampling

- A forma que faremos data sampling é com uma sliding window que vai pegar um texto e associar cada nova palavra a uma "resposta" cuja "pergunta" é o texto até antes dessa palavra.

- Um dataloader é um objeto que deve ter operações que facilitem o carregamento de dados para seu modelo. Basicamente ele tem várias amostras e organiza elas em batches, onde seu dataset todo será alimentado dessa forma. As amostras devem ter mesmo tamanho, batches não precisam. Em especial, um dataloader tem que ser capaz de criar iteradores ou iterar sobre batches com um for in, por exemplo. Iteradores geralmente podem ser usados com next() e inicializados com iter().

- Para fazermos um data loader eficiente para a LLM, criamos tensores com os equivalentes de [a, b, c, ..., y] e [b, c, ..., z] como input e output, pois nos fornece uma associação entre quais inputs correspondem a um output. Nosso data loader terá:
    - Uma memória (max_length) de quantos tokens ele se lembra no máximo;
    - Um stride definindo o passo entre as janelas;
    - A cada janela, pegamos as sequências [A -> B], [A, B -> C], [A, B, C -> D], ..., [A, B, C, ..., Y -> Z] para considerar como respostas e perguntas. Isso é representado por dois tensores [A, B, ..., Y] e [B, C, ..., Z] sendo adicionados a input e output ids.
    - Drop last para descartar a última janela se ela for menor, pois nesses tensores ai vão ser feitas operações matriciais.
    - Num workers para controlar quantos núcleos de processamento faremos multithreading.

- Para criar o dataloader, primeiro fazemos uma classe para nosso dataset que herda de Dataset do torch.utils.data. Ela deve ter uma inicialização

### Embedding

- O embedding é o processo de mapear os tokens a vetores. A ideia inicial é que queremos associar palavras distintas a vetores distintos, mas no fim, queremos que palavras parecidas sejam vetores mais próximos, enquanto palavras distantes sejam vetores distantes. Vetores com ângulos pequenos são palavras que frequenetemente aparecem em contextos similares, ortogonais são contextos independentes e vetores com ângulos acima de reto indicam correlação negativa, isto é, palavras que aparecem em contextos mutuamente exclusivos, de certa forma.

- Para fazer isso, criaremos uma camada de embedding do Pytorch e durante o treinamento, ele aprenderá os pesos corretos, que são as coordenadas dos vetores. Elas são inicializadas como valores aleatórios, que em média vão gerar vetores quase ortogonais pela maldição da dimensionalidade.
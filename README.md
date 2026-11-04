# Heston Hedging Lab

Código de pesquisa da dissertação sobre **hedge de carteiras de opções com volatilidade estocástica, aprendizado por reforço e aprendizado de operadores**.

## Qual é a ideia?

Na etapa atual, investigamos **transferência de políticas de hedge entre regimes de volatilidade estocástica**, mantendo fixa a composição da carteira e os três instrumentos negociáveis. A infraestrutura anterior de carteiras variáveis permanece disponível, mas não integra o experimento principal.

A pesquisa seguirá este caminho:

**Simular Heston sob Q → precificar nove calls fixas → validar a contabilidade dos três forwards → treinar especialistas SAC-Trade → testar transferência entre valores de `xi`.**

A intenção não é escolher antecipadamente uma rede neural vencedora. Mais adiante, vamos comparar modelos simples com arquiteturas capazes de trabalhar com informações funcionais, como superfícies de volatilidade implícita.

## Experimento atual: carteira fixa e três forwards sob Heston-Q

A segunda linha experimental usa **`fixed_eniac_book()`**, com nove calls
**vendidas** (strikes 96, 100, 104; vencimentos 42, 84, 126 dias úteis).
Ela **não substitui** as carteiras variáveis e os baselines spot anteriores.

Os parâmetros do piloto estão no bloco `forward_experiment` de `config.json`:
medida **Q**, `kappa=1`, `theta=0.2` (variância de longo prazo), `v0=0.04`,
`rho=+0.7`, `xi=0.2`, 250 dias úteis e taxa contínua `log(1.10)`.
Os três instrumentos de hedge são *forwards sintéticos*, com preços
`F(t,T) = S_t exp((r-q)(T-t))`. A posição gera uma diferença diária de preços,
creditada ao caixa. **É uma convenção simplificada de marcação diária**,
não um modelo completo dos fluxos de um forward OTC; reproduz a estrutura
comparativa do ENIAC e não simula margem de futuros.

No novo experimento todas as trajetórias e marcações são sob **Q**.
As métricas de cauda resultantes são, portanto, estatísticas **sob Q**, não
estimativas de frequência de perdas sob a probabilidade física P.
O código antigo `evaluate.py` e as demonstrações antigas de `demo.ipynb`
ainda possuem configurações P para os baselines históricos; **não integram**
este experimento Q de forwards.

A inicialização é delta-neutral **nas unidades do forward**:
`q_j(0) = clip(-Delta_book,j / (dF_j/dS), 0, 3)`.
É ligeiramente diferente de copiar diretamente o delta spot do ENIAC,
pois o próprio preço forward varia com S. O custo da abertura é contabilizado.
Cada ação é o incremento solicitado `Delta q` em `[-1,1]^3`; após os limites
`[0,3]`, negociações inferiores a 0,01 não são executadas.
O custo em cada execução é `half_spread * forward_mid * |Delta q| + 0.5`,
com metade do spread de 1%, 2% e 3% por vencimento. No vencimento, a
posição é forçosamente encerrada e paga custos, conforme a convenção do ENIAC.
A ação executada em t já afeta o P&L de t para t+1.

A implementação financeira está em `hedging.py`:
`forward_prices()`, `initial_forward_positions()`,
`execute_forward_trades()` e `simulate_forward_hedge()`.
**Ainda não há ambiente Gymnasium, política SAC, recompensa ou treinamento.**
É possível executar uma avaliação de ações prescritas, por exemplo:

```python
import json
import numpy as np
from heston import HestonParams, MarketDynamics, simulate_heston
from portfolio import fixed_eniac_book
from hedging import value_book_paths, initial_forward_positions, simulate_forward_hedge

cfg = json.load(open("config.json"))["forward_experiment"]
p = HestonParams(**cfg["heston"])
m = MarketDynamics(cfg["measure"], cfg["spot0"],
                   cfg["risk_free_rate"], cfg["dividend_yield"])
paths = simulate_heston(p, m, n_paths=1, n_steps=cfg["n_steps"],
                        trading_days_per_year=cfg["trading_days_per_year"],
                        seed=cfg["seed"])
book = fixed_eniac_book()
marks = value_book_paths(book, paths, p, m,
                         trading_days_per_year=cfg["trading_days_per_year"])
initial = initial_forward_positions(book, paths, p, m,
                                    expiries_days=tuple(cfg["expiries_days"]),
                                    trading_days_per_year=cfg["trading_days_per_year"])
result = simulate_forward_hedge(
    paths, marks, m, initial_positions=initial,
    actions=np.zeros((1, cfg["n_steps"], 3)),
    expiries_days=tuple(cfg["expiries_days"]),
    spread_half_pcts=tuple(cfg["spread_half_pcts"]),
    fixed_cost=cfg["fixed_cost"], min_trade=cfg["min_trade"],
    max_trade_increment=cfg["max_trade_increment"],
    max_position=cfg["max_position"],
    trading_days_per_year=cfg["trading_days_per_year"],
)
print(result.terminal_pnl, result.transaction_cost)
```

Esse exemplo representa **manter o hedge inicial sem rebalanceamento voluntário**;
não demonstra a superioridade de uma estratégia. Execute os testes com
`python -m pytest -q` antes de alterar a dinâmica financeira.

## O que o código já faz?

### 1. Simula o mercado (`heston.py`)

Utilizamos o **modelo de Heston** para gerar possíveis trajetórias de um ativo financeiro. Diferentemente do modelo Black–Scholes mais simples, em Heston a volatilidade não é constante: ela muda ao longo do tempo e pode se relacionar com as oscilações do preço.

Para cada trajetória simulada, guardamos:

- o preço do ativo em cada dia;
- sua **variância instantânea** (a volatilidade é a raiz quadrada dessa variância);
- o tempo correspondente a cada observação.

A simulação usa passos diários e uma aproximação numérica chamada *full-truncation Euler* para a variância. O script `simulate.py` executa um exemplo sob a **medida neutra ao risco (Q)**. O script `evaluate.py` simula **cenários físicos (P)** separadamente e precifica as opções usando os parâmetros sob **Q**. Ambos os conjuntos de parâmetros são explícitos no `config.json`; no exemplo inicial são iguais, mas não precisam ser.

### 2. Cria carteiras diferentes (`portfolio.py`)

Em vez de trabalhar sempre com uma lista fixa de opções, geramos carteiras com diferentes quantidades de contratos, strikes, vencimentos e posições.

O código também calcula o **pagamento de cada opção quando ela vence**. Por exemplo: uma posição comprada de duas calls com strike 100 paga 20 se o ativo estiver em 110 no vencimento (duas vezes a diferença de 10).

**Pagamento no vencimento não é preço da opção.** O pagamento depende somente do preço no vencimento; o preço teórico antes do vencimento é calculado separadamente em `pricing.py`. Os pagamentos alimentam o caixa da simulação de hedge em `hedging.py`. O experimento usa o preço teórico inicial como prêmio, sem spread na negociação dos próprios contratos.

### 3. Calcula preços e sensibilidades (`pricing.py`)

Utilizamos a **fórmula semi-analítica de Heston** para calcular quanto vale, no modelo, uma call ou put europeia antes de vencer. Diferentemente dos pagamentos no vencimento, o preço depende também do tempo restante, da variância atual e das condições de mercado.

O módulo calcula **delta** (sensibilidade ao preço do ativo), **gamma** (como o delta muda) e sensibilidade à **variância instantânea**. Também informa a sensibilidade à raiz da variância, mas **ela não é o mesmo que a vega em relação à volatilidade implícita de mercado**. Essas sensibilidades são aproximadas por diferenças finitas e ainda não foram otimizadas para milhões de avaliações.

Podemos calcular o valor agregado de uma carteira inteira, considerando posições compradas e vendidas. Um contrato que já venceu sai da avaliação; seu pagamento deve ser contabilizado separadamente no caixa.

**Os preços exigem parâmetros sob a medida neutra ao risco (Q).** Uma simulação de trajetórias sob P, quando implementada para avaliar hedge, não pode ser usada como se fosse uma parametrização Q de precificação.

### 4. Constrói a superfície de volatilidade implícita (`volatility.py`)

A **volatilidade implícita** é a volatilidade que, inserida na fórmula de
Black–Scholes, reproduz o preço de uma opção. Ela serve para expressar preços
de opções em uma escala comparável, mesmo quando seus strikes ou vencimentos
são diferentes.

Nosso programa calcula um preço por Heston e encontra numericamente a
volatilidade equivalente em Black–Scholes. Repetindo esse processo para
vários strikes e vencimentos, obtemos uma **superfície de volatilidade
implícita (IV)**. Essa superfície descreve as condições de mercado do modelo,
**não os contratos específicos de uma carteira**.

A grade é definida em `config.json` por vencimentos e *log-moneyness*
`log(K / F)`, onde `F` é o preço a termo do ativo. Assim, ela pode ser
reutilizada para carteiras diferentes. Na configuração inicial usamos 9 pontos
de moneyness e 7 vencimentos (63 volatilidades implícitas).

O programa salva tanto a superfície inicial quanto **uma superfície em cada
dia de cada trajetória**. Para não repetir milhões de vezes a integração
numérica de Heston, pré-calculamos superfícies para alguns níveis de variância
(`variance_nodes` em `config.json`) e interpolamos entre eles. Como as
coordenadas usam `log(K/F)` e os parâmetros Q são constantes nesta
simulação, a volatilidade implícita depende da variância atual, não do nível
absoluto do preço do ativo.

Os vencimentos dessa grade são **prazos restantes constantes** (21 dias,
42 dias, etc.) em cada data; não representam a evolução de uma mesma opção
que se aproxima do vencimento. Em cada instante, os strikes da superfície
acompanham o preço a termo do ativo. Para reconstruí-los:
`K = S_t * exp((r - q) * prazo_em_anos + log_moneyness)`.

O valor interpolado é aproximado e só é calculado dentro dos limites da grade
de variâncias. Se a simulação ultrapassar esses limites, o programa sinaliza
o problema em vez de extrapolar silenciosamente. São superfícies *teóricas
do modelo Heston*, e não cotações observadas em bolsa.

### 5. Executa e salva os experimentos (`simulate.py`)

Esse script reúne as partes anteriores: lê os parâmetros, simula trajetórias, gera carteiras, calcula seus valores e sensibilidades iniciais sob Q, constrói superfícies IV ao longo das trajetórias e registra os pagamentos contratuais em cada trajetória.

Com a configuração inicial em `config.json`, ele gera **1.000 trajetórias de 126 dias úteis e 16 carteiras**, cujos tamanhos possíveis são 1, 3, 5 ou 9 opções.

Os arquivos ficam em `results/simulations/`:

| Arquivo | O que contém |
|---|---|
| `paths.npz` | Preços, variâncias e instantes das trajetórias simuladas |
| `portfolios.jsonl` | Lista de contratos de cada carteira |
| `expiry_cashflows.npz` | Pagamentos de cada carteira ao longo das trajetórias |
| `initial_prices.jsonl` | Preço teórico inicial de cada carteira e sensibilidades, quando a configuração está sob Q |
| `implied_vol_surface.npz` | Grade e superfície IV no estado inicial sob Q |
| `implied_vol_paths.npz` | Superfícies IV para cada trajetória/dia, grade de prazos/moneyness e tabela usada na interpolação sob Q |
| `summary.json` | Parâmetros utilizados e diagnósticos básicos da simulação |

As sementes aleatórias em `config.json` permitem reproduzir os resultados. Os arquivos de `results/` não são versionados no Git.

### 6. Avalia carteiras diariamente e faz hedge (`hedging.py`, `evaluate.py`)

Para avaliar uma estratégia, precisamos saber quanto vale a carteira **em cada dia**, não somente na abertura. `hedging.py` usa os preços e as sensibilidades sob Q para reavaliar as opções que ainda não venceram. Quando um contrato vence, seu valor sai da carteira e o pagamento entra **uma única vez no caixa**.

Em paralelo, mantemos um número de ações do ativo subjacente e uma conta de caixa. Ao negociar ações, o caixa diminui ou aumenta pelo valor negociado e pelos custos. Entre dois dias, o caixa rende à taxa livre de risco e a posição recebe dividendos quando houver. A riqueza total reúne **carteira de opções + ações + caixa**. O caixa começa com o negativo do valor teórico da carteira, financiando os contratos e estabelecendo riqueza inicial zero antes dos custos de hedge. O resultado final considera o encerramento da posição em ações.

O script `evaluate.py` compara quatro estratégias transparentes, ainda **sem RL**:

- **Sem hedge:** não negocia ações;
- **Delta diário:** ajusta a posição para neutralizar o delta todos os dias;
- **Delta periódico:** faz esse ajuste a cada 21 dias úteis;
- **Banda fixa:** só negocia quando a posição sai de uma faixa ao redor do hedge delta, movendo-a até o limite da banda.

Os custos incluem uma parcela proporcional ao volume e uma parcela fixa por negociação; negócios abaixo de `min_trade` não são executados. O resultado é medido pelo P&L final, perdas extremas (`q95` e `CVaR95`), custos, volume de negociação e quantidade de operações. **Essas estratégias não são necessariamente ótimas**: são pontos de referência para avaliar futuramente o RL.

Executando:

```bash
python evaluate.py
# Um piloto menor:
python evaluate.py --n-paths 32 --n-books 2
```

Os arquivos ficam em `results/hedging/`:

| Arquivo | Conteúdo |
|---|---|
| `market_paths.npz` | Preços e variâncias sob P, além das superfícies IV calculadas sob Q nos mesmos estados |
| `portfolios.jsonl` | Carteiras avaliadas |
| `book_paths.npz` | Valores, delta, gamma, sensibilidade à variância e pagamentos de cada carteira por dia |
| `hedging_results.npz` | P&L, custos, operações e volume por trajetória e estratégia |
| `metrics.json` | Médias e medidas de risco das estratégias, além dos parâmetros P e Q |

**Limitação numérica:** a avaliação em lote usa uma integração de Fourier mais rápida do que a integração adaptativa do `pricing.py`. Em opções muito afastadas do preço de exercício, com prazo curto e volatilidade efetiva muito pequena, o código usa o valor intrínseco descontado como aproximação, quando o valor temporal é esperado ser desprezível. Isso exige validação adicional antes de usar essas regiões em experimentos científicos definitivos. O primeiro piloto tem poucas trajetórias; seus quantis de risco **não são estimativas estatisticamente robustas**.

## O que vem depois?

**Hedge com RL:** o próximo passo é construir um ambiente Gymnasium a partir das trajetórias **Q-Heston**, da carteira fixa de nove calls e da contabilidade já testada dos três forwards sintéticos. A primeira política será **SAC-Trade**, com ações de negociação vetoriais. O SAC-Band não integra o primeiro experimento.

**Aprendizado e generalização:** investigar se modelos contextuais ou operadores neurais conseguem transferir políticas para **valores de `xi` não vistos no treinamento**, sem alterar a carteira fixa nem treinar um especialista para cada regime de teste. O desempenho será avaliado pelo risco financeiro, custos e eficiência computacional, não apenas pelo erro de predição.

Essas etapas são a direção da pesquisa, **não funcionalidades que já existem no repositório**.

## Onde encontrar cada coisa?

| Arquivo ou pasta | Função |
|---|---|
| `heston.py` | Modelo e simulação das trajetórias |
| `portfolio.py` | Contratos, carteiras e pagamentos no vencimento |
| `pricing.py` | Preços Heston, sensibilidades e valor de carteiras |
| `volatility.py` | Inversão da volatilidade implícita e construção da superfície IV |
| `hedging.py` | Valores diários, contabilidade do hedge e políticas de referência |
| `evaluate.py` | Compara políticas de hedge em trajetórias físicas P |
| `simulate.py` | Execução do experimento atual |
| `config.json` | Parâmetros das simulações e sementes |
| `tests/` | Testes de consistência numérica e dos contratos |
| `results/` | Resultados gerados pelas execuções |

## Como executar?

Na pasta do repositório, na primeira utilização (Linux/macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python simulate.py
python evaluate.py
python -m pytest -q
```

Nas próximas utilizações:

```bash
source .venv/bin/activate
python simulate.py
```

Para alterar a quantidade de trajetórias ou carteiras, edite `config.json`. Para testar rapidamente outro tamanho sem alterar o arquivo:

```bash
python simulate.py --n-paths 100 --n-books 4
```

A pasta `.venv/` contém somente o ambiente Python local e também não deve ser versionada.

## Aprender a usar os módulos — notebook demonstrativo

Abra **`demo.ipynb`** no JupyterLab ou no VS Code, selecione o Python do
`.venv` e execute as células em ordem. Cada seção mostra **como chamar as
funções existentes**, quais argumentos recebem, o que retornam e como
interpretar os gráficos gerados diretamente no notebook.

O notebook percorre `heston.py`, `portfolio.py`, `pricing.py`, `volatility.py`
e `hedging.py`, com exemplos pequenos que você pode modificar livremente.
Não reimplementa os modelos nem cria novos resultados de pesquisa.

Para abrir no JupyterLab, caso ainda não esteja instalado:

```bash
source .venv/bin/activate
python -m pip install jupyterlab
jupyter lab demo.ipynb
```

No VS Code, basta abrir `demo.ipynb` e selecionar o kernel do `.venv`.
Os gráficos aparecem **no próprio notebook**. Os experimentos completos
continuam em `simulate.py` e `evaluate.py`.

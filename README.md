# Heston Hedging Lab

Código de pesquisa da dissertação sobre **hedge de carteiras de opções com volatilidade estocástica, aprendizado por reforço e aprendizado de operadores**.

**Git e execução no servidor:** veja [docs/GIT_SERVER.md](docs/GIT_SERVER.md)
para publicar somente código/configurações, recriar os splits no servidor e
executar o SAC com **30 ambientes `SubprocVecEnv`**, como no ENIAC. Os arquivos
experimentais em `results/` não devem ser versionados.

## Qual é a ideia?

Na etapa atual, investigamos **transferência de políticas de hedge entre regimes de volatilidade estocástica**, mantendo fixa a composição da carteira e os três instrumentos negociáveis. A infraestrutura anterior de carteiras variáveis permanece disponível, mas não integra o experimento principal.

A pesquisa seguirá este caminho:

**Simular Heston sob Q → precificar nove calls fixas → validar a contabilidade dos três forwards → validar o ambiente Gymnasium → treinar especialistas SAC-Trade → testar transferência entre valores de `xi`.**

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
**O ambiente Gymnasium está implementado em `rl_env.py`; o piloto de treinamento SAC-Trade foi acrescentado em `train_sac_trade.py`.**
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

## Ambiente SAC-Trade: integração financeira (sem treinamento)

O módulo `rl_env.py` oferece duas interfaces para uma mesma transição financeira:

- `ForwardEpisode`: núcleo de episódio que não depende do Gymnasium, útil para
  verificações financeiras e experimentos prescritos.
- `HestonForwardGymEnv`: adaptador compatível com `gymnasium.Env` que expõe
  `reset`, `step`, `action_space` e `observation_space`. **Não treina** um SAC.

Ambos reutilizam `advance_forward_hedge()` de `hedging.py`, também chamada
por `simulate_forward_hedge()`. Isso impede que o simulador e o futuro agente
mantenham duas versões independentes das regras de negócio.

**Observação do piloto:**
`[spot, variância instantânea, 3 vencimentos restantes (anos), 3 posições forward, 9 preços das calls]`,
com dimensão **17**. A carteira contém nove opções vendidas, mas os nove preços
observados são valores positivos **por contrato**, antes do sinal da posição.
Para reproduzir as **16 observações** do ENIAC, configure
`include_variance=false` no bloco `forward_experiment`. A variância Heston
é conhecida pelo simulador, mas *não* é uma observação de mercado diretamente
acessível; trata-se de uma hipótese explícita de observabilidade. A superfície
IV ainda **não** foi adicionada às observações do RL.

**Ação:** três incrementos em `[-1, 1]`; o executor impõe `[0,3]` para as
posições, o mínimo negociável e custos por instrumento. A ação executada no
dia `t` já recebe o resultado financeiro de `t` a `t+1`. No vencimento,
o instrumento é encerrado, com seu respectivo custo.

**Recompensa:** se `p_t` for a variação **líquida de riqueza** no passo,
`reward_t = p_t - c * p_t**2`. O primeiro passo inclui o custo de constituição
do hedge inicial. `quadratic_reward_penalty` está **provisoriamente em zero**;
assim, a soma das recompensas coincide com o P&L terminal. O parâmetro
`c` precisará ser calibrado antes de comparar políticas. Não confundir essa
recompensa com otimização direta de CVaR.

As trajetórias são pré-simuladas integralmente sob Q e a carteira é marcada
uma vez antes do episódio; o agente recebe apenas **informações no instante
atual**. Essa forma está adequada à validação do ambiente. Um pequeno pool
fixo de trajetórias, entretanto, **não é um pipeline suficiente para treinar
especialistas**: antes do SAC precisaremos definir regeneração, semente,
amostragem, custo computacional e conjuntos independentes de avaliação.

```bash
# Verificação financeira sem dependência do Gymnasium
python rl_env.py --n-paths 2 --seed 303

# Testes, incluindo equivalência entre rollout e avaliador
python -m pytest -q
```

Após instalar `gymnasium` pelo `requirements.txt`, o adaptador pode ser
instanciado assim:

```python
from rl_env import prepare_forward_episodes, HestonForwardGymEnv
pool, cfg = prepare_forward_episodes("config.json", n_paths=2, seed=303)
env = HestonForwardGymEnv(pool, cfg)
obs, info = env.reset(seed=303, options={"path_index": 0})
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

**Atenção:** os campos `total_transaction_cost`, `pnl_step` e `wealth` no `info`
expõem a contabilidade verificada, mas o retorno financeiro final só aparece
como `terminal_pnl` no último passo. Valores de treinamento, política RL e
resultados de generalização não são produzidos nesta etapa.

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

**Hedge com RL:** o ambiente Gymnasium e o piloto SAC-Trade agora possuem fluxo de treinamento e avaliação. O SAC-Band será incluído como segunda parametrização, com as mesmas referências financeiras e dados de validação.

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
| `rl_env.py` | Núcleo de episódio causal e adaptador Gymnasium para SAC-Trade |
| `train_sac_trade.py` | Piloto SAC-Trade: treinamento em chunks Q, seleção de checkpoint e avaliação |
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

## Experimentos reproduzíveis: trajetórias Q e baselines antes do SAC

`forward_experiments.py` prepara conjuntos independentes de **trajetórias completas**
(treino, validação, teste) sob Q, sempre mantendo carteira e regime Heston do
bloco `forward_experiment`. As sementes são explícitas e distintas; não dividimos
aleatoriamente as linhas de uma mesma trajetória entre os conjuntos.

```bash
# Execução piloto pequena, para verificar o pipeline — não para publicar CVaR
python forward_experiments.py prepare --config config.json \
    --out-dir results/forward_experiments/pilot_xi_0p20 \
    --train-paths 64 --validation-paths 32 --test-paths 32

# Comparar regras transparentes SOMENTE na validação enquanto calibramos
python forward_experiments.py baseline --config config.json \
    --out-dir results/forward_experiments/pilot_xi_0p20 \
    --split validation

# Gerar coeficientes candidatos para penalização quadrática — sem escolher um
python forward_experiments.py calibrate --config config.json \
    --out-dir results/forward_experiments/pilot_xi_0p20
```

A chamada `prepare` grava `train.npz`, `validation.npz` e `test.npz` com
seus arquivos JSON de proveniência, um `manifest.json`, SHA-256 dos arquivos e
impressão digital da **configuração financeira**. A leitura exige a mesma
configuração e verifica a integridade do NPZ. Para alterar tamanho, regime ou
sementes, escolha **outro diretório**; não há sobrescrita silenciosa.

As regras de referência são: `no_hedge` (sem constituição inicial),
`hold_initial` (manter hedge inicial), `delta_daily`, `delta_periodic_21`
e `delta_band_0p5`. A última negocia até a **borda** da banda, não até o
centro. Todas as posições, limites de incrementos, custos e liquidações são
processados pela contabilidade já testada em `hedging.py`. Os deltas por
vencimento são produzidos junto dos preços Heston, evitando reprecificar
cada opção em cada decisão. Políticas que dependem de delta consultam somente
o **dia presente**. Não existe SAC neste módulo.

Os resumos são salvos em `validation_baselines.csv` / `.json`, e os resultados
por trajetória em `.npz`. `calibrate` calcula o percentil 90 de
`abs(P&L líquido do passo)` dos baselines `hold_initial` e `delta_daily`
**exclusivamente na validação** e produz candidatos `c = f / p90`, para
`f = 0, 0.05, 0.1, 0.2`. O critério corresponde a uma penalidade quadrática
cuja magnitude relativa, naquele valor de referência, é `f`. **Não seleciona
um c ótimo**; o valor final deverá ser escolhido comparando desempenho e
risco em validação, antes de consultar o teste.

Em experimentos longos, um pool fixo pequeno pode provocar sobreajuste.
Para futura integração com o treinador, `iter_train_chunks()` gera **novos
blocos reproduzíveis de trajetórias Q**, com sementes derivadas de
`base_seed` e do índice do bloco, sem carregar todos na memória nem
reutilizar validação e teste. O treinador `train_sac_trade.py` passa a consumir esses blocos na execução piloto.

**Limitações:** CVaR95 estimado com 32 trajetórias não é estatisticamente
confiável; os tamanhos acima são somente para verificação operacional.
Resultados reais precisam de amostras maiores, múltiplas sementes de treino,
intervalos de incerteza e um teste final reservado. Todas as métricas deste
experimento são **sob Q**. O protocolo de partição entre regimes (`xi`
observado versus não observado) será especificado separadamente depois dos
especialistas.

## Piloto SAC-Trade (Heston-Q, primeira arquitetura)

A dissertação utilizará **SAC-Trade e SAC-Band**; começamos pelo SAC-Trade
por decisão metodológica, não por exclusão do SAC-Band. Este script não é o
código de treinamento do ENIAC: reutiliza o ambiente financeiro validado.

Após `python forward_experiments.py prepare ...` e a avaliação dos baselines:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python train_sac_trade.py train \
  --splits-dir results/forward_experiments/pilot_xi_0p20 \
  --out-dir results/sac_trade/pilot_c2_limit1_seed123 \
  --timesteps 3000 --eval-every 1500 --train-chunk-paths 16 \
  --penalty 2.0 --max-trade-increment 1.0 --seed 123
python train_sac_trade.py evaluate \
  --splits-dir results/forward_experiments/pilot_xi_0p20 \
  --run-dir results/sac_trade/pilot_c2_limit1_seed123
```

Amostragem de treino: novos *chunks* Q com sementes separadas; cada trajetória
é selecionada uma vez por chunk. Seleção de checkpoints: **média da recompensa
bruta por episódio na validação**. As métricas de P&L, q95 e CVaR95 são
registradas separadamente, sem confundir objetivo RL e risco de cauda.
O teste não é consultado no treinamento e `evaluate --split test` exige a
confirmação explícita `--allow-test`, para usar apenas após fixar escolhas.

A ação é o incremento nos três forwards, com limites originais `[-1,1]^3`
e posição `[0,3]^3`. Para o controle de capacidade de rebalanceamento,
`--max-trade-increment 3` permite atingir qualquer posição válida em uma
única decisão. **O SAC-Band é implementado em `train_sac_band.py`**, com os mesmos
instrumentos, observações comparáveis, orçamento e avaliação. O teste de
`c=2` vem antes da análise de `c` calibrado; não comparar números de
recompensa entre valores de `c`, comparar métricas financeiras sob Q.

A cada avaliação, o treinador também salva `step_000000005000.zip` e
`step_000000005000.vecnormalize.pkl` (e equivalentes para os demais passos).
Isso permite reavaliar qualquer checkpoint intermediário, por exemplo:

```bash
python train_sac_trade.py evaluate \
  --splits-dir results/forward_experiments/pilot_xi_0p20 \
  --run-dir results/sac_trade/pilot_c002635_limit1_30k \
  --model step_000000005000
```

Os checkpoints continuam sendo **selecionados por recompensa média na validação**;
CVaR95 e demais métricas permanecem diagnósticos. A nova opção
`--no-save-eval-checkpoints` desativa as cópias intermediárias em execuções
muito longas. Os arquivos `best` e `final` são sempre preservados.

`results/sac_trade/.../run.json` registra parâmetros, semente e fingerprint
financeiro; `validation_history.jsonl` contém avaliações; arquivos
`best.zip` / `best.vecnormalize.pkl` e equivalentes `final` preservam
checkpoint e normalização correspondentes. `validation_best_evaluation.json`
e `.npz` guardam métricas e resultados individuais. Os 3000 passos são
somente um teste funcional, **não** um treinamento conclusivo.

Por padrão o piloto SAC-Trade recebe 20 observações: as 17 anteriores
mais os **três deltas atuais por vencimento**, calculados sob Q. Isso evita
que SAC-Band receba informação exclusiva quando introduzirmos seu prior de
delta. Para ablação, use `--no-include-bucket-deltas`; essa opção não altera
a versão de 17 entradas do ambiente didático anterior.

Hiperparâmetros piloto por padrão: `gamma=0.995`, `tau=0.002`,
`learning_rate=1e-4`, `batch_size=256`, rede `[128,128]`, próximos às
referências do ENIAC. Orçamento e quantidade de ambientes ainda são
reduzidos para a validação funcional; **não constitui reprodução dos 50M**
de passos nem prova de equivalência de desempenho.

## Piloto SAC-Band (Heston-Q): largura de banda em torno do delta dos forwards

`train_sac_band.py` reutiliza **o mesmo SAC, motor financeiro, Heston-Q, separação
train/validation/test, observação de 20 componentes, estrutura de rede,
normalização, custo de abertura e função de recompensa** de SAC-Trade.
A ação `b_t` contém três **semilarguras** não negativas em `[0,1.5]`.
Para cada bucket `j`, o alvo de hedge atual é o delta da carteira de opções
sob Q convertido para **unidades do forward**:

`q_delta[j] = clip(-option_bucket_delta[j] / exp((r-q)*tau[j]), 0, 3)`.

A banda é `[q_delta-b, q_delta+b]`: se a posição está dentro, não negocia;
caso contrário, negocia até a **fronteira mais próxima**, respeitando `[0,3]`
e o mínimo de negociação `0.01`. Os contratos que vencem são encerrados pelo
motor financeiro compartilhado. SAC-Band pode negociar até **3 unidades de uma
vez**, ao contrário da configuração SAC-Trade `[-1,1]`: essa comparação inicial
**não isola exclusivamente o efeito da arquitetura**. Uma comparação com
capacidade de execução igual será considerada depois, se for necessária.
O sinal do delta é importante: o livro é composto por opções **vendidas**.

Primeiro executar apenas o teste de integração (`3k` passos):

```bash
python -m pytest -q
python train_sac_band.py train \
  --splits-dir results/forward_experiments/pilot_xi_0p20 \
  --out-dir results/sac_band/smoke_c2_seed123_3k \
  --timesteps 3000 --eval-every 1500 --train-chunk-paths 16 \
  --penalty 2.0 --max-band-width 1.5 --seed 123
python train_sac_band.py evaluate \
  --splits-dir results/forward_experiments/pilot_xi_0p20 \
  --run-dir results/sac_band/smoke_c2_seed123_3k --model best
```

Após validar a integração, executar o diagnóstico comparável de 30 mil passos:

```bash
python train_sac_band.py train \
  --splits-dir results/forward_experiments/pilot_xi_0p20 \
  --out-dir results/sac_band/diagnostic_c2_seed123_30k \
  --timesteps 30000 --eval-every 5000 --train-chunk-paths 16 \
  --penalty 2.0 --max-band-width 1.5 --seed 123
python train_sac_band.py evaluate \
  --splits-dir results/forward_experiments/pilot_xi_0p20 \
  --run-dir results/sac_band/diagnostic_c2_seed123_30k --model best
python train_sac_band.py evaluate \
  --splits-dir results/forward_experiments/pilot_xi_0p20 \
  --run-dir results/sac_band/diagnostic_c2_seed123_30k --model final
```

Em `validation_..._evaluation.npz`, `actions` significa **larguras de
banda solicitadas**, enquanto `executed_change` guarda as mudanças de posição
realmente executadas no rebalanceamento; não confundir as duas grandezas.
Em SAC-Trade, `actions` continua significando incrementos solicitados.
Não usar `--split test` para ajuste de modelos. **32 trajetórias não permitem
conclusões confiáveis sobre CVaR95**; estes são diagnósticos, não resultados
científicos finais. Por enquanto o coeficiente é **c=2**, sem nova calibração.

### Training progress and ETA (SAC-Trade and SAC-Band)

Both training scripts display elapsed wall-clock time, steps per second, and an
**estimated** remaining time. The reporter counts validation and checkpoint
saving overhead when estimating throughput; generating fresh training chunks
also contributes to elapsed time. Predictions may fluctuate, especially early
on or when chunk preparation/validation is expensive.

Set the reporting cadence with `--progress-every` (default: 5000). For example:

```bash
python train_sac_band.py train \
  --splits-dir results/forward_experiments/pilot_xi_0p20 \
  --out-dir results/sac_band/diagnostic_c2_seed123_30k \
  --timesteps 30000 --eval-every 5000 --progress-every 1000 \
  --penalty 2.0 --max-band-width 1.5 --seed 123
```

The console prints lines such as
`[progress] 5,000/30,000 (16.7%) | elapsed 00:01:45 | 47.6 steps/s | ETA ~00:08:45`.
These numbers are *illustrative*, not benchmarks. Every report is also saved to
`training_progress.jsonl` in the run directory (see with `tail -f`). The training
summary includes `training_wall_seconds`. This does not change the environment,
policy, financial metrics or model-selection rule.

For a very large run, choose a coarser cadence, e.g. `--progress-every 100000`.
Be careful with `--eval-every` and `--save-eval-checkpoints`: with 50 million
steps and evaluations every 5000 steps, the default could create **10,000**
full model/normalization snapshots, consuming substantial disk space. Use an
appropriate evaluation cadence and, if desired, `--no-save-eval-checkpoints`
(best/final checkpoints are still saved). Time estimates are not a substitute
for checkpoint/restart support, which is not implemented by this pilot trainer.

### Constant-band control before expanding Heston regimes

`regime_experiments.py` provides two *pretraining* checks. It imports the
existing `ForwardBandEpisode`, therefore a fixed width uses exactly the same
position target, trading costs, minimum trade, forced expiry close and reward
accounting as the trained SAC-Band. This is **not** the older
`delta_band_0p5` baseline, which restricts daily increments to one unit.

**A. Compare an exploratory constant width 0.75 with the saved SAC-Band 30k
checkpoint on the same 32 validation paths** (no new training):

```bash
python regime_experiments.py control \
  --config config.json \
  --splits-dir results/forward_experiments/pilot_xi_0p20 \
  --out-dir results/controls/xi_0p20_band_0p75 \
  --width 0.75 \
  --compare-evaluation results/sac_band/diagnostic_c2_seed123_30k/validation_best_evaluation.npz
```

Outputs: `validation_constant_band.json` (metrics and paired comparison) and
`validation_constant_band.npz` (individual trajectories, actions, executed
increments). The width was motivated by the observed short-run learned
policy and is **not** independently calibrated: interpret this as a diagnostic,
not confirmatory evidence. No test paths are opened.

**B. Prepare separated Q-Heston regime datasets**, changing only `xi` in
`forward_experiment.heston` (all nine options, costs, other parameters and
observation conventions are unchanged):

```bash
python regime_experiments.py prepare \
  --base-config config.json \
  --out-root results/forward_experiments/regime_pilots \
  --xi 0 --xi 0.20 --xi 0.40 \
  --train-paths 64 --validation-paths 128 --test-paths 32
```

The three configurations and independent datasets are saved under
`results/forward_experiments/regime_pilots/xi_0p00/`, `xi_0p20/` and
`xi_0p40/`, each containing `config.json`, `regime.json` and `splits/`.
The three split seeds are 1101, 2202 and 3303 within each regime; using the
same seed *across different xi configurations* provides common random numbers
but does not mix training and validation. Pre-existing regime folders are never
overwritten. `xi=0` is deterministic **variance**, not constant variance.

Example small SAC-Band integration run after preparing `xi=0`:

```bash
python train_sac_band.py train \
  --config results/forward_experiments/regime_pilots/xi_0p00/config.json \
  --splits-dir results/forward_experiments/regime_pilots/xi_0p00/splits \
  --out-dir results/sac_band/xi_0p00_smoke_3k \
  --timesteps 3000 --eval-every 1500 --progress-every 500 \
  --train-chunk-paths 16 --penalty 2.0 --max-band-width 1.5 --seed 123
```

The same flags work for `xi_0p20` and `xi_0p40` when each path is replaced.
A subsequent ~100k-steps-per-regime screening (and later multiple seeds,
validation >=1000 paths) should be authorized by the learning curves, not
mistaken for a 50-million-step confirmatory study. Keep all held-out `test`
paths untouched until architectures, reward and model-selection rules are frozen.


### SAC-Band execution rule — returns to delta reference (ENIAC-compatible)

**New rule:** each width action defines a no-trade zone around the current
**forward-position delta target**. Stay put when inside the band (including
its edges). Outside, execute a full rebalance **to the target**, not to the
nearest boundary; apply the existing minimum-trade, transaction-cost,
position-limit and maturity rules. This is the rule described in the ENIAC
poster. The old Heston-Band experiments applied a different *boundary* rule.

New band runs and evaluations carry `"band_execution": "delta_center"` in
`run.json` / evaluation JSON, and new constant-band controls record the same
field. **Do not re-evaluate earlier SAC-Band checkpoints with this change:**
those checkpoints were trained under boundary execution, and the evaluator
rejects them explicitly. All existing JSON/NPZ results remain intact.
SAC-Trade is unchanged. New SAC-Band models must be **trained again**.

The constant-width control on the 128 frozen validation scenarios of each
regime does *not* require an RL checkpoint and may be run immediately,
with output stored separately from previous boundary-rule controls. From
the repository root:

```bash
ROOT=results/forward_experiments/regime_pilots
for XI in xi_0p00 xi_0p20 xi_0p40; do
  python regime_experiments.py control \
    --config "$ROOT/$XI/config.json" \
    --splits-dir "$ROOT/$XI/splits" \
    --out-dir "results/controls/${XI}_center_0p75_128" \
    --width 0.75
done
```

For example the two results for `xi=0.20` are saved under
`results/controls/xi_0p20_center_0p75_128/validation_constant_band.json`
and `results/controls/xi_0p20_center_0p75_128/validation_constant_band.npz`.
All controls only use the `validation` split. Do **not** append
`--compare-evaluation` pointing to a pre-patch SAC-Band checkpoint.

## Treinamento de longa duração — Heston-Q (SAC-Trade / SAC-Band)

O executor independente `train_sac_scale.py` prepara **ambas** as arquiteturas para
execuções com milhões de transições. Os pilotos (`train_sac_trade.py`,
`train_sac_band.py`) continuam disponíveis para reprodução histórica, mas **não
selecionam o checkpoint pelo protocolo ENIAC**. Para os novos especialistas,
use **somente** `train_sac_scale.py`:

- **Recompensa:** incremento de P&L líquido menos `2 * (incremento)^2`, normalizada
  pelo `VecNormalize` durante o aprendizado. **Seleção:** menor Q95 de perda
  líquida terminal em validação; empate mantém o checkpoint anterior. CVaR95,
  P&L, custos e negociações são métricas de diagnóstico, não de seleção.
- **SAC-Trade:** ações em `[-1,1]^3`. **SAC-Band:** larguras em `[0,1.5]^3`,
  com **retorno ao delta-alvo** (`delta_center`); essa é a regra escolhida para
  Heston, não a variante Band-Boundary do script ENIAC anexado. Ambos observam os mesmos 20
  indicadores financeiros. A contabilidade financeira e os parâmetros Heston-Q
  existentes não são alterados.
- **Treino vetorizado:** `--n-envs` (referência 30), `--vec-env subproc|dummy`;
  cada worker tem stream Q exclusivo, gerado por chunks e nunca lê validação/teste.
  `--timesteps` conta transições somadas dos ambientes. O alvo pode ser
  arredondado para cima por até `n_envs-1` transições, como no SB3.
- **Recuperação:** snapshots completos em `OUT/recovery/step_*/`, com
  `model.zip`, `replay_buffer.pkl`, `vecnormalize.pkl`, `streams.pkl`,
  `rng.pkl` e metadados. `OUT/recovery/latest.json` aponta apenas para um
  snapshot integralmente gravado. Duas gerações são mantidas; `--resume` usa
  exclusivamente a mais recente. Uma execução normal não sobrescreve arquivos.
  **Limite de reprodutibilidade:** a retomada não é garantida bit a bit devido
  à ordem de updates internos e resets SB3; é adequada à continuidade
  estatística de longas execuções e registra essa limitação.
- **Checkpoint para inferência:** `OUT/best.zip` e
  `OUT/best.vecnormalize.pkl` são selecionados pelo Q95, e cada avaliação
  guarda `OUT/step_*.zip` + snapshot de normalização correspondente. Os
  checkpoints de recuperação são independentes dos checkpoints para avaliação.
  `--resume-every` e `--eval-every` podem ter cadências diferentes.
- **Monitoramento:** `OUT/training_progress.jsonl` contém velocidade,
  porcentagem, tempo decorrido e ETA. `OUT/validation_history.jsonl` registra
  cada checkpoint validado, seu Q95 e demais métricas. Resultados finais
  em `OUT/summary.json`.

### Verificação operacional (sem valor científico)

Da raiz do repositório, usando o split-piloto já existente (128 trajetórias):

```bash
ROOT=results/forward_experiments/regime_pilots/xi_0p20
RUN=results/sac_trade/scale_smoke_resume_xi_0p20
COMMON="--action-mode trade --config $ROOT/config.json --splits-dir $ROOT/splits \
  --out-dir $RUN --timesteps 6000 --n-envs 2 --vec-env dummy \
  --eval-every 1500 --resume-every 1000 --progress-every 500 \
  --learning-starts 256 --buffer-size 5000 --batch-size 256"
python train_sac_scale.py train $COMMON --stop-after 3000
python train_sac_scale.py train $COMMON --resume
python train_sac_scale.py evaluate --run-dir "$RUN" --model best
```

O primeiro comando deve **parar intencionalmente** sem `summary.json`, mas com
snapshot completo em `RUN/recovery/`. O segundo retoma do snapshot e conclui;
o terceiro verifica que `best` é avaliável com as mesmas configurações e dados.
O diretório precisa estar vazio antes do primeiro comando. Não reexecute o
primeiro comando sobre dados existentes.

### Preparar conjuntos independentes em escala

**Atenção:** gerar 5.000 trajetórias de validação + 20.000 de teste pode
consumir tempo e armazenamento consideráveis. Fazer apenas depois do smoke.

```bash
python forward_experiments.py prepare \
  --config results/forward_experiments/regime_pilots/xi_0p20/config.json \
  --out-dir results/forward_experiments/xi_0p20_scale_splits \
  --train-paths 64 --validation-paths 5000 --test-paths 20000
```

Esses conjuntos são gravados uma vez; a geração de episódios de **treino**
para o SAC é independente do arquivo `train.npz`. O arquivo `test.npz` ficará
reservado e não será acessado durante a seleção por Q95. Para uso em outros
regimes, gerar uma configuração e splits separados por `xi`.

### Primeiro treinamento longo (após o smoke e a preparação de 5.000 cenários)

```bash
python train_sac_scale.py train --action-mode trade \
  --config results/forward_experiments/regime_pilots/xi_0p20/config.json \
  --splits-dir results/forward_experiments/xi_0p20_scale_splits \
  --out-dir results/sac_trade/xi_0p20_scale_seed123 \
  --timesteps 50000000 --n-envs 30 --vec-env subproc \
  --eval-every 2500000 --resume-every 500000 --progress-every 50000 \
  --train-chunk-paths 16 --learning-starts 50000 --buffer-size 500000 \
  --penalty 2.0 --max-trade-increment 1.0 --seed 123
```

Para retomar, **repetir exatamente esses argumentos e acrescentar `--resume`**.
Não alterar o código, a configuração, as seeds nem a quantidade de workers
entre a execução original e a retomada. Avaliação:

```bash
python train_sac_scale.py evaluate \
  --run-dir results/sac_trade/xi_0p20_scale_seed123 --model best
```

O comando de avaliação lê a configuração e os splits congelados do `run.json`;
não há risco de omitir `--config` / `--splits-dir` por descuido. Para `SAC-Band`,
substituir `--action-mode trade` por `band`, usar `--max-band-width 1.5` e
**outro** `--out-dir`. O conjunto de teste requer autorização explícita
`--split test --allow-test`, **somente após congelar as escolhas científicas**.

**Critérios de liberação dos 50M:** smoke com `--stop-after`/`--resume` no
ambiente que efetivamente possui Stable-Baselines3, Q95 registrado como seletor,
checksums preservados, tempo/memória/disk budget aferidos. O teste de integração
não substitui os 50M nem busca demonstrar convergência.


### Compatibilidade do primeiro piloto de retomada (correção 2026-11)

A primeira versão de `train_sac_scale.py` comparava diretamente os campos de
`run.json` (listas) com tuplas do dataclass Python. Assim, a retomada podia
falhar com `Resume configuration mismatch for config` mesmo usando exatamente
as flags originais. O patch de compatibilidade compara representações JSON
canônicas e possui `scale_resume_compat.json`, que permite **somente** a
transição entre as duas assinaturas completas de código auditadas, sem alterar
`run.json` nem os snapshots em `recovery/`. Mudanças futuras de código ou de
configuração financeira continuam impedindo a retomada. Preserve o manifest no
repositório e não edite seus hashes manualmente.


### Correção de retomada: observação terminal no VecEnv (2026-11-05)

O primeiro snapshot real de 3.000 transições foi feito no **dia 114/126** de
um episódio ativo. O modelo SB3 salvo ainda continha `_last_obs`; com
`SAC.load(force_reset=False)`, `learn(reset_num_timesteps=False)` não fazia
`VecEnv.reset()` no começo da retomada. O antigo `_restore_on_reset` acabava
entregando a observação **terminal** quando o `DummyVecEnv` reiniciava o
ambiente após o vencimento, causando `episode terminated; call reset()`.

**Regra corrigida de retomada:** carregar o SAC com `force_reset=True` e
iniciar **um novo episódio por worker** no primeiro `VecEnv.reset()`. O episódio
incompleto do momento do snapshot é descartado; os exemplos já coletados
continuam no replay buffer, e o stream prossegue na **próxima trajetória Q
não utilizada**. Normalização e recompensas acumuladas do novo episódio são
reinicializadas pelo `VecNormalize.reset()`. Portanto, a retomada preserva a
continuidade *estatística* do treinamento, mas não é bit a bit idêntica ao
processamento sem interrupções. Os contadores de passos não retrocedem.

O arquivo `scale_resume_compat.json` autoriza explicitamente as assinaturas
das duas versões legadas a utilizar esta correção, mantendo as proteções de
mudança de configuração, dados, sementes e código. Não edite checkpoints,
hashes ou o manifest à mão. Os testes de regressão simulam um episódio
interrompido e o reinício subsequente no `DummyVecEnv` real (quando a
biblioteca SB3 estiver instalada). Antes de autorizar 50 milhões de passos,
**retomar e concluir o smoke operacional existente de 3.000 para 6.000**.

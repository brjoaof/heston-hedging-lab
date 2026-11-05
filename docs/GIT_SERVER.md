# Git e execução no servidor — especialistas Heston-Q (ENIAC/30 envs)

Este repositório versiona **código-fonte, configuração financeira, testes e
instruções**. **Não** versiona `results/`, conjuntos `.npz`, modelos `.zip`,
replay buffers, snapshots de recuperação ou o ambiente `.venv/`.

Os scripts originais do ENIAC foram usados para conferir os parâmetros de
*treinamento*. **O script Band-Boundary anexado não define a execução do novo
SAC-Band**: em Heston, a regra definitiva é `delta_center` (dentro da banda,
não negociar; fora, rebalancear ao delta-alvo dos forwards).

## 1. Preparar e publicar o código na máquina local

Na raiz do projeto:

```bash
python scripts/setup_gitignore.py  # preserva qualquer .gitignore existente
python -m pytest -q
git status --short
git check-ignore -v results/qualquer_experimento/model.zip
git add -A
git diff --cached --stat
```

Antes do commit, confira se `git diff --cached --name-only` contém apenas
código, testes, documentos e configurações. Especialmente, **não devem
aparecer** `results/`, `.venv/`, `.npz`, `.pkl`, `.zip` ou dados privados.

O script `scripts/setup_gitignore.py` é idempotente: acrescenta um bloco
identificado ao `.gitignore` existente (ou cria um se não houver) e não
sobrescreve outras regras do projeto. **Inclua o `.gitignore` resultante no
commit.**

**Importante:** `.gitignore` não remove do índice arquivos já rastreados.
Se algum resultado tiver sido adicionado anteriormente, use **somente se
`git ls-files results/` não estiver vazio**:

```bash
git rm -r --cached -- results/
```

A opção `--cached` mantém os dados no seu disco e remove apenas do índice Git.
Depois revise novamente `git diff --cached --stat`.

```bash
git commit -m "Prepare Heston SAC scale for ENIAC 30-env server runs"
git push origin HEAD
```

Se o repositório não tiver um remoto, configure previamente a URL real do seu
servidor (`git remote -v`; `git remote add origin URL_DO_REPOSITORIO`).
Não inclua tokens e senhas na URL versionada.

## 2. Clonar/atualizar no servidor

```bash
git clone URL_DO_REPOSITORIO heston-hedging-lab
cd heston-hedging-lab
# Nos acessos seguintes: git pull --ff-only
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

A versão exata de Python/PyTorch/SB3/Gymnasium e os controles do experimento
são registrados no `run.json` quando o treinamento começa. Para comparações
reprodutíveis, mantenha o mesmo ambiente de software entre sementes.

## 3. Recriar os dados da validação/teste no servidor (ξ=0,20)

O arquivo **`config.json` é rastreado pelo Git**. O bloco
`forward_experiment` contém a configuração financeira do especialista Heston-Q
em ξ=0,20. Os arquivos binários de trajetória são gerados **no servidor**:

```bash
python forward_experiments.py prepare \
  --config config.json \
  --out-dir results/forward_experiments/xi_0p20_scale_splits \
  --train-paths 64 --validation-paths 5000 --test-paths 20000 \
  --train-seed 1101 --validation-seed 2202 --test-seed 3303
```

Este procedimento gera `manifest.json` com sementes e hashes. O SAC usa
**streams de treinamento independentes, gerados sob demanda**; o `train.npz`
preparado não é o dataset de treinamento SAC. A validação seleciona o
checkpoint pelo menor Q95. O teste de 20 mil caminhos permanece intocado até
a seleção final.

Para outros valores de ξ, use `regime_experiments.py prepare` com o mesmo
`config.json` versionado e valores explícitos de `--xi`; salve cada regime
em um diretório diferente dentro de `results/`.

## 4. Treinamento longo: configuração do ENIAC (não executar antes do smoke)

O executor `train_sac_scale.py` é comum às duas arquiteturas e tem como
configuração de referência **30 workers em `SubprocVecEnv`**, 50 milhões de
transições somadas, `train_freq=(1, "step")`, `gradient_steps=1`,
`learning_starts=50_000`, `buffer_size=500_000`, `batch_size=256`,
`gamma=0.995`, `tau=0.002`, `learning_rate=1e-4`, entropia automática,
`torch.set_num_threads(1)`, normalização de observação/recompensa e rede
`[128,128]`. Por vetorizar em grupos de 30, o SB3 pode avançar até
**50.000.010 transições** para satisfazer o orçamento de 50 milhões.

Exemplo do primeiro especialista SAC-Trade com `--seed 1`:

```bash
python train_sac_scale.py train \
  --action-mode trade --config config.json \
  --splits-dir results/forward_experiments/xi_0p20_scale_splits \
  --out-dir results/sac_trade/xi_0p20_seed1_50M \
  --timesteps 50000000 --n-envs 30 --vec-env subproc \
  --eval-every 2500000 --resume-every 500000 --progress-every 50000 \
  --train-chunk-paths 16 --learning-starts 50000 --buffer-size 500000 \
  --batch-size 256 --gamma 0.995 --tau 0.002 --learning-rate 0.0001 \
  --penalty 2.0 --max-trade-increment 1.0 \
  --seed 1 --train-seed 551177
```

SAC-Band deve usar `--action-mode band --max-band-width 1.5` e **outro
`--out-dir`**. A regra `delta_center` já está no `band_policy.py` e é
verificada ao avaliar checkpoints.

Para retomar, execute **exatamente o mesmo comando**, adicionando `--resume`.
Não altere configuração, número de workers, dados ou código em um treinamento
em andamento. A retomada é estatisticamente consistente, mas não bit a bit
idêntica a uma execução ininterrupta.

Avaliação final, depois de congelar o checkpoint selecionado:

```bash
python train_sac_scale.py evaluate \
  --run-dir results/sac_trade/xi_0p20_seed1_50M --model best
# SOMENTE após seleção:
# python train_sac_scale.py evaluate \
#   --run-dir results/sac_trade/xi_0p20_seed1_50M \
#   --model best --split test --allow-test
```

## 5. Critério para liberar o treinamento grande

No próprio servidor, executar uma única verificação operacional curta com
**30 subprocessos** e interrupção/retomada. Sua finalidade é testar
multiprocessing, normalização, integridade da recuperação e consumo de
CPU/memória/disco — **não** avaliar a qualidade financeira em poucos passos.
A validação de 5.000 trajetórias pode ser custosa; para o smoke, use os
splits-piloto ou um conjunto de validação pequeno gerado separadamente.

**Não copie diretórios `recovery/` para o Git**. Eles possuem caminhos e
versões de software associados à execução de origem. Para treinar no
servidor, faça um novo `run` com a versão de código e ambiente instalada lá.

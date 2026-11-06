# Filas sequenciais de especialistas Heston-Q

## Execução em dois terminais (Trade / Band)

**Não executar antes do smoke de 30 workers.** Para evitar seleção inadvertida de
regimes de teste do operador, o manifesto versionado
`configs/expert_schedule.json` define duas listas **disjuntas**:

- `regimes`: especialistas usados como candidatos a dados de treinamento do
  operador (`0.00`, `0.10`, `0.20`, `0.30`, `0.40`).
- `reserved_operator_regimes`: regimes de avaliação de generalização
  (`0.05`, `0.15`, `0.25`, `0.35`). Nenhum especialista destes regimes é
  treinado pela fila inicial. Se forem necessários oráculos específicos para
  comparação posterior, produzi-los **separadamente** e nunca usar suas ações
  para ajustar o operador.

O manifesto propõe **cinco sementes** por regime e arquitetura, seguindo a
prática de replicação do ENIAC. Isto corresponde a **25 treinamentos por fila,
50 no total**, ou **2,5 bilhões de transições**. É um orçamento COMPUTACIONAL
muito grande, não um pequeno smoke test. Revise o plano e os recursos do
servidor antes de executar; a lista e as sementes são configuráveis.

### Preparar os dados uma vez, antes de abrir as duas filas

```bash
python scripts/expert_queue.py prepare --dry-run
python scripts/expert_queue.py prepare
python scripts/expert_queue.py status
```

Este preparo gera um conjunto imutável de 64/5.000/20.000 trajetórias por
regime, usando as sementes previamente fixadas. O treinador consome validação
para seleção do Q95; o teste permanece reservado. O próprio script detecta
regimes já preparados e não sobrescreve datasets; um preparo incompleto exige
inspeção manual, nunca exclusão automática.

### Prévia do agendamento, sem iniciar nenhum treino

```bash
python scripts/expert_queue.py run --mode trade
python scripts/expert_queue.py run --mode band
```

### Rodar com persistência após logout

Em **um terminal**, dentro da sessão `tmux`, execute:

```bash
tmux new -s sac-trade
# dentro da sessão:
source .venv/bin/activate
python scripts/expert_queue.py run --mode trade --execute
```

Em **outro terminal**:

```bash
tmux new -s sac-band
# dentro da sessão:
source .venv/bin/activate
python scripts/expert_queue.py run --mode band --execute
```

Use `Ctrl+B`, depois `D`, para desconectar sem interromper o treinamento.
Reabra com `tmux attach -t sac-trade` ou `tmux attach -t sac-band`.
Logs individuais ficam em `results/queue_logs/{trade,band}/`.

O script executa cada regime/semente **sequencialmente por arquitetura**,
verifica `summary.json` antes de pular um experimento concluído, retoma um
experimento incompleto **apenas** se houver `recovery/latest.json`, e **para a
fila ao primeiro erro**, preservando os resultados. Para continuar depois de
uma interrupção, reexecute **o mesmo comando da fila**, no mesmo commit e
ambiente, sem mudar o manifesto nem o código do treinador. Não há mudanças
silenciosas nos arquivos existentes.

**Capacidade:** as duas filas simultâneas com 30 workers cada requerem até
60 subprocessos de ambiente, além dos processos principais. Em um servidor de
52 CPUs lógicas/60 GiB essa configuração pode causar oversubscription e
pressão de RAM. O uso simultâneo exige medição prévia de CPU, RAM e throughput;
se não houver capacidade, execute as filas em sequência ou aloque mais
recursos, preservando 30 ambientes **por treinamento**.

**Importante sobre Git:** `results/`, incluindo logs, datasets e checkpoints,
continua excluído do Git. Não use `git pull` para modificar código durante a
execução dos especialistas: o treinador verifica a assinatura do código ao
retomar um checkpoint.

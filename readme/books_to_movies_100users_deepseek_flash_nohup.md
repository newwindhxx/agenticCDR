# Books → Movies：100 用户 DeepSeek Flash 实验命令

本文记录本次 AgentCF++ 实验的启动、后台运行、进度检查、断点续跑和结果查看命令。

## 实验配置

| 配置项 | 取值 |
| --- | --- |
| 源域 | `Books` |
| 目标域 | `Movies_and_TV` |
| Profile | `full` |
| 用户数 | 100 |
| 训练交互数 | 1824 |
| 每个评测样本的候选数 | 10（1 个正样本、9 个负样本） |
| LLM | `deepseek-flash` |
| 思考模式 | 关闭 |
| 群体记忆 | 启用 |
| 配置文件 | `configs/books_to_movies.yaml` |

以下命令均从项目根目录执行：

```bash
cd /opt/data/private/nf/data/agenticCDR
source .venv/bin/activate
```

## 检查预计调用量

此命令只读取本地数据，不会调用 DeepSeek API：

```bash
python scripts/estimate_run.py \
  --config configs/books_to_movies.yaml \
  --profile full
```

当前数据的估算结果为：训练请求 7296 次、验证和测试排序请求 200 次、兴趣标签请求 100 次、群体命名请求最多 10 次，总计最多 7606 次。

## 设置 API Key

API Key 通过环境变量传入，不写入 YAML、日志或 Git：

```bash
read -rsp "请输入 DeepSeek API Key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
```

检查格式，但不打印 Key：

```bash
python - <<'PY'
import os

key = os.environ.get("DEEPSEEK_API_KEY", "")
assert key, "DEEPSEEK_API_KEY 没有设置"
assert key.isascii(), "API Key 包含非 ASCII 字符"
assert not any(c.isspace() for c in key), "API Key 包含空格或换行"
print("API Key 格式检查通过")
PY
```

## 使用 nohup 启动完整实验

下面的任务会顺序完成：

1. 训练 100 用户的用户和物品记忆。
2. 在验证集上执行无群体记忆评测。
3. 从训练记忆构建群体共享记忆。
4. 在测试集上执行有群体记忆评测。

```bash
RUN_ID="books_to_movies_full_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "runs/$RUN_ID"

export RUN_ID
LOG_FILE="runs/$RUN_ID/nohup-full.log"

nohup bash -c '
set -euo pipefail

cd /opt/data/private/nf/data/agenticCDR
source .venv/bin/activate

echo "===== 1/4 开始训练 ====="
python scripts/train_agentcfpp.py \
  --config configs/books_to_movies.yaml \
  --profile full \
  --run-id "$RUN_ID" \
  --resume

echo "===== 2/4 验证集评测 ====="
python scripts/evaluate_agentcfpp.py \
  --config configs/books_to_movies.yaml \
  --profile full \
  --split validation \
  --run-id "$RUN_ID"

echo "===== 3/4 构建群体记忆 ====="
python scripts/build_group_memory.py \
  --config configs/books_to_movies.yaml \
  --profile full \
  --run-id "$RUN_ID"

echo "===== 4/4 测试集群体记忆评测 ====="
python scripts/evaluate_agentcfpp.py \
  --config configs/books_to_movies.yaml \
  --profile full \
  --split test \
  --use-group-memory \
  --run-id "$RUN_ID"

echo "===== FULL EXPERIMENT COMPLETE ====="
' > "$LOG_FILE" 2>&1 < /dev/null &

JOB_PID=$!
echo "$JOB_PID" > "runs/$RUN_ID/full.pid"

echo "RUN_ID=$RUN_ID"
echo "PID=$JOB_PID"
echo "日志=$LOG_FILE"

disown "$JOB_PID" 2>/dev/null || true
```

`nohup` 可以防止关闭终端或断开 SSH 后任务退出。服务器重启、容器停止、内存不足或手动终止进程仍会中断任务。

## 找到当前 full 运行

新开终端后执行：

```bash
cd /opt/data/private/nf/data/agenticCDR
source .venv/bin/activate

RUN_ID=$(cat runs/.latest_books_to_movies_full)
echo "$RUN_ID"
```

后续命令都使用这个 `RUN_ID`。

## 查看实时日志

```bash
tail -f "runs/$RUN_ID/nohup-full.log"
```

按 `Ctrl+C` 只会退出日志查看，不会停止后台实验。

查看最近 100 行：

```bash
tail -100 "runs/$RUN_ID/nohup-full.log"
```

只查看关键进度和阶段切换：

```bash
grep -E 'completed=|Training complete|Evaluation|开始训练|验证集评测|构建群体记忆|测试集群体记忆评测|FULL EXPERIMENT' \
  "runs/$RUN_ID/nohup-full.log" | tail -50
```

## 查看进程状态

```bash
PID=$(cat "runs/$RUN_ID/full.pid")
ps -p "$PID" -o pid,etime,%cpu,%mem,cmd
```

如果 `ps` 只显示标题而没有进程记录，说明该进程已经结束。此时检查日志末尾判断是正常完成还是发生错误：

```bash
tail -100 "runs/$RUN_ID/nohup-full.log"
```

## 精确查看训练完成进度

检查点在每条完整训练交互结束后原子保存。下面的命令显示已完成条数：

```bash
python - "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

run_id = sys.argv[1]
state_path = Path("runs") / run_id / "memory" / "state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
completed = int(state.get("last_completed", -1)) + 1
total = 1824
print(f"训练进度: {completed}/{total} ({completed / total:.2%})")
PY
```

也可以查看日志里的最后一条训练记录：

```bash
grep 'completed=' "runs/$RUN_ID/run.log" | tail -1
```

## 查看 Token 使用量

每次成功调用的 Token 使用量保存在 `llm_cache.jsonl`。汇总当前消耗：

```bash
python - "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

run_id = sys.argv[1]
cache_path = Path("runs") / run_id / "llm_cache.jsonl"
records = []
if cache_path.exists():
    with cache_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

prompt = sum((row.get("usage") or {}).get("prompt_tokens") or 0 for row in records)
completion = sum((row.get("usage") or {}).get("completion_tokens") or 0 for row in records)
total = sum((row.get("usage") or {}).get("total_tokens") or 0 for row in records)

print(f"成功请求数: {len(records)}")
print(f"输入 Token: {prompt:,}")
print(f"输出 Token: {completion:,}")
print(f"总 Token: {total:,}")
PY
```

## 中断后断点续跑

重新进入项目并再次输入 API Key：

```bash
cd /opt/data/private/nf/data/agenticCDR
source .venv/bin/activate

read -rsp "请输入 DeepSeek API Key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY

RUN_ID=$(cat runs/.latest_books_to_movies_full)
export RUN_ID
LOG_FILE="runs/$RUN_ID/nohup-full.log"
```

然后用相同的 `RUN_ID` 重新运行完整流程：

```bash
nohup bash -c '
set -euo pipefail
cd /opt/data/private/nf/data/agenticCDR
source .venv/bin/activate

python scripts/train_agentcfpp.py \
  --config configs/books_to_movies.yaml \
  --profile full \
  --run-id "$RUN_ID" \
  --resume

python scripts/evaluate_agentcfpp.py \
  --config configs/books_to_movies.yaml \
  --profile full \
  --split validation \
  --run-id "$RUN_ID"

python scripts/build_group_memory.py \
  --config configs/books_to_movies.yaml \
  --profile full \
  --run-id "$RUN_ID"

python scripts/evaluate_agentcfpp.py \
  --config configs/books_to_movies.yaml \
  --profile full \
  --split test \
  --use-group-memory \
  --run-id "$RUN_ID"

echo "===== FULL EXPERIMENT COMPLETE ====="
' >> "$LOG_FILE" 2>&1 < /dev/null &

JOB_PID=$!
echo "$JOB_PID" > "runs/$RUN_ID/full.pid"
echo "已恢复运行：RUN_ID=$RUN_ID PID=$JOB_PID"
disown "$JOB_PID" 2>/dev/null || true
```

训练阶段会从 `memory/state.json` 中的下一条交互继续。已经成功返回的 LLM 请求会从 `llm_cache.jsonl` 读取，不会重复计费。

## 判断完整实验是否成功

```bash
grep -F 'FULL EXPERIMENT COMPLETE' "runs/$RUN_ID/nohup-full.log"
```

如果输出 `FULL EXPERIMENT COMPLETE`，说明四个阶段均已成功结束。

检查主要结果文件：

```bash
ls -lh \
  "runs/$RUN_ID/metrics.json" \
  "runs/$RUN_ID/predictions_validation_nogroup.jsonl" \
  "runs/$RUN_ID/predictions_test_group.jsonl" \
  "runs/$RUN_ID/memory/groups.json" \
  "runs/$RUN_ID/llm_cache.jsonl"
```

## 查看最终指标

查看完整指标：

```bash
cat "runs/$RUN_ID/metrics.json"
```

格式化输出主要指标：

```bash
python - "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

run_id = sys.argv[1]
metrics_path = Path("runs") / run_id / "metrics.json"
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

for evaluation_id, result in metrics.items():
    print(f"\n[{evaluation_id}]")
    for key in (
        "valid_samples",
        "parse_failures",
        "MRR",
        "NDCG@1",
        "NDCG@5",
        "NDCG@10",
        "average_target_rank",
    ):
        print(f"{key}: {result.get(key)}")
PY
```

本流程正常完成后，`metrics.json` 至少应包含：

- `validation_nogroup`：验证集、不使用群体记忆。
- `test_group`：测试集、使用群体记忆。

## 追加无群体记忆测试结果

如果还需要比较 AgentCF++ 开启和关闭群体记忆的差异，可以追加测试集无群体记忆评测：

```bash
read -rsp "请输入 DeepSeek API Key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY

RUN_ID=$(cat runs/.latest_books_to_movies_full)
NO_GROUP_LOG="runs/$RUN_ID/nohup-test-nogroup.log"

nohup python scripts/evaluate_agentcfpp.py \
  --config configs/books_to_movies.yaml \
  --profile full \
  --split test \
  --run-id "$RUN_ID" \
  > "$NO_GROUP_LOG" 2>&1 < /dev/null &

echo $! > "runs/$RUN_ID/test-nogroup.pid"
echo "日志=$NO_GROUP_LOG"
```

查看这项评测：

```bash
tail -f "runs/$RUN_ID/nohup-test-nogroup.log"
```

完成后，`metrics.json` 会新增 `test_nogroup`，对应的逐样本预测保存在 `predictions_test_nogroup.jsonl`。

## 结果目录结构

```text
runs/<run_id>/
├── config.resolved.yaml
├── data_manifest.json
├── full.pid
├── llm_cache.jsonl
├── metrics.json
├── nohup-full.log
├── predictions.jsonl
├── predictions_validation_nogroup.jsonl
├── predictions_test_group.jsonl
├── run.log
├── checkpoints/
└── memory/
    ├── state.json
    └── groups.json
```

其中：

- `run.log`：程序记录的阶段和交互进度。
- `nohup-full.log`：标准输出、进度条和异常堆栈。
- `llm_cache.jsonl`：LLM 响应缓存及 Token 用量。
- `memory/state.json`：训练后的用户和物品记忆检查点。
- `memory/groups.json`：群体共享记忆。
- `predictions_*.jsonl`：逐用户候选排序结果。
- `metrics.json`：汇总指标和逐用户目标物品排名。

#!/bin/bash

# =============================
# 参数检查
# =============================
if [ $# -ne 3 ]; then
    echo "Usage: $0 <example.json> <output.json> <NUM>"
    exit 1
fi

EXAMPLE_JSON=$1
OUTPUT_JSON=$2
NUM=$3

# =============================
# 校验示例文件
# =============================
if [ ! -f "$EXAMPLE_JSON" ]; then
    echo "Error: example json not found: $EXAMPLE_JSON"
    exit 1
fi

# =============================
# 若输出文件存在，先删除
# =============================
if [ -f "$OUTPUT_JSON" ]; then
    echo "Output file exists, removing: $OUTPUT_JSON"
    rm -f "$OUTPUT_JSON"
fi

# =============================
# 生成 analyseConditions
# =============================
TASKS=$(jq -c --argjson num "$NUM" '
[
  range(1; $num + 1) as $i
  | .
  | .standName = (.standName | sub("[0-9]+$"; "") + ("000" + ($i|tostring))[-3:])
  | .taskID    = (.taskID    | sub("[0-9]+$"; "") + ("000" + ($i|tostring))[-3:])
  | .taskName  = (.taskName  | sub("[0-9]+$"; "") + ("000" + ($i|tostring))[-3:])
  | .deviceInfo.deviceName
      = (.deviceInfo.deviceName | sub("[0-9]+$"; "") + ($i|tostring))
  | .deviceInfo.deviceCode
      = (.deviceInfo.deviceCode | sub("[0-9]+$"; "") + ("00000000" + ($i|tostring))[-8:])
  | .deviceInfo.sourceRTSP
      = (.deviceInfo.sourceRTSP | sub("mystream[0-9]+$"; "mystream" + ($i|tostring)))
]
' "$EXAMPLE_JSON")

# =============================
# 输出最终 JSON
# =============================
jq -n --argjson tasks "$TASKS" '
{
  analyseConditions: $tasks
}
' > "$OUTPUT_JSON"

echo "Generated $NUM tasks -> $OUTPUT_JSON"
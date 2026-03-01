#!/bin/bash
# 指定使用 bash 解释器来执行此脚本

# 后台启动脚本
# 用法: bash start.sh run4.sh 或 bash /workspace/backdoor-toolbox-new/run/start.sh /workspace/backdoor-toolbox-new/run/run5.sh
# 检查进程是否还在运行:
#   ps aux | grep run4.sh
#   ps aux | grep run5.sh
#
# 查看日志确认还在运行:
#   tail -f run4.log
#   tail -f run5.log

# 切换到脚本工作目录，确保在正确的路径下执行
cd /workspace/backdoor-toolbox-new

# 检查脚本参数：$# 表示传入的参数个数，-lt 1 表示小于1（即没有参数）
if [ $# -lt 1 ]; then
    # 如果没有传入参数，显示用法说明
    echo "用法: bash start.sh <脚本名>"
    # 显示使用示例
    echo "示例: bash start.sh run4.sh"
    # 以错误码1退出脚本，表示执行失败
    exit 1
fi

# 获取第一个参数（传入的脚本名称，如 run4.sh），保存到变量 SCRIPT_NAME
SCRIPT_NAME="$1"
# 生成日志文件名：去掉 .sh 后缀，加上 .log 后缀（如 run4.sh -> run4.log）
LOG_NAME="${SCRIPT_NAME%.sh}.log"
# 生成PID文件名：去掉 .sh 后缀，加上 .pid 后缀（如 run4.sh -> run4.pid）
PID_FILE="${SCRIPT_NAME%.sh}.pid"

# 使用 nohup 在后台运行脚本：
#   nohup: 即使终端关闭，进程也继续运行
#   bash "$SCRIPT_NAME": 执行传入的脚本
#   > "$LOG_NAME": 将标准输出重定向到日志文件
#   2>&1: 将标准错误也重定向到标准输出（即日志文件）
#   &: 在后台运行
nohup bash "$SCRIPT_NAME" > "$LOG_NAME" 2>&1 &

# $! 是上一个后台进程的进程ID，将其保存到PID文件中
echo $! > "$PID_FILE"
# 读取PID文件并显示，告知用户脚本已启动及其进程ID
echo "脚本已在后台启动，PID: $(cat $PID_FILE)"
# 提示用户如何查看日志文件
echo "查看日志: tail -f $LOG_NAME"


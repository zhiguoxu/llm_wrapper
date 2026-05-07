import sys
import json
from pathlib import Path

from loguru import logger as logru_logger



def project_patcher(record):
    """
    补丁函数：每一条日志产生时都会调用它。
    在这里从 ContextVars 中提取信息，注入到 record["extra"] 中。
    """


def serialize_json(record):
    """
    针对 ELK 等系统的结构化日志处理
    """
    subset = {
        "time": record["time"].strftime("%Y-%m-%d %H:%M:%S.%f"),
        "level": record["level"].name,
        "msg": record["message"],
        "device_sn": record["extra"].get("device_sn"),
        "file": f"{record['file'].name}:{record['line']}",
        "module": record["module"],  # 模块名
        "function": record["function"],  # 函数名
    }
    # 将除了标准字段外的所有 extra 字段也塞进去
    subset.update({k: v for k, v in record["extra"].items() if k not in subset})
    return json.dumps(subset, ensure_ascii=False)


# 核心配置类
class LogManager:
    @staticmethod
    def setup(
            log_dir: str = "logs",
            level: str = "INFO",
            show_console: bool = True,
            async_mode: bool = True,
            json_output: bool = True
    ):
        # 移除 Loguru 默认的 handler
        logru_logger.remove()

        # 基础格式 (类似你之前的 || 风格)
        base_fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <6}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<magenta>{extra[device_sn]}</magenta> | "
            "<level>{message}</level>"
        )

        def json_format(record):
            json_str = serialize_json(record)
            return json_str.replace("{", "{{").replace("}", "}}").replace("<", r"\<") + "\n"

        # A. 控制台输出
        if show_console:
            logru_logger.add(
                sys.stdout,
                level=level,
                format=base_fmt,
                colorize=True,
                enqueue=async_mode,
                backtrace=True,
                diagnose=True
            )

        # B. 滚动文件输出 (INFO)
        log_path = Path(log_dir).resolve()
        log_path.mkdir(parents=True, exist_ok=True)

        # json 格式，方便日志处理
        logru_logger.add(log_path / "server.log",
                   level=level,
                   format=json_format if json_output else base_fmt,
                   rotation="00:00",  # 每天零点翻转
                   retention="30 days",  # 保留30天
                   compression="zip",  # 压缩旧日志
                   enqueue=async_mode,
                   encoding="utf-8")

        # C. 错误日志(仅ERROR和CRITICAL)
        logru_logger.add(log_path / "error.log",
                   level="ERROR",  # 只有级别 >= ERROR 的日志才会写入此文件
                   # 错误日志可以改用更易读的文本格式，也可以继续用 JSON
                   format=json_format if json_output else base_fmt,
                   rotation="1 week",  # 错误日志通常较少，可以按周滚动
                   retention="1 months",
                   enqueue=async_mode,
                   encoding="utf-8",
                   backtrace=True,  # 错误日志额外记录堆栈信息
                   diagnose=True  # 记录变量值
                   )

        # 核心：注入 Patch
        return logru_logger.patch(project_patcher)


# 初始化全局
logger = LogManager.setup(level='INFO')

if __name__ == "__main__":
    logger.info("第一条日志")

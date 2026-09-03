"""
Shared Utilities and Helper Functions
Preventable Readmissions Risk Stratifier Pipeline
"""

import os
import sys
import time
import yaml
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from functools import wraps

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Engine


def setup_logger(
    name: str = "readmission_logger",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configures and returns a thread-safe structured logger.
    
    Parameters
    ----------
    name : str
        Name of the logger.
    log_file : Optional[str]
        Optional destination file path for persistent logs.
    level : int
        Logging level (default: logging.INFO).
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = setup_logger(__name__)


def load_config(config_path: Union[str, Path] = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads and parses the YAML configuration file.
    
    Parameters
    ----------
    config_path : Union[str, Path]
        Path to YAML configuration file.
        
    Returns
    -------
    Dict[str, Any]
        Parsed dictionary of configuration options.
    """
    path = Path(config_path)
    if not path.is_file():
        # Fallback check relative to repository root if executed from subfolder
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / "config" / "config.yaml"
        if candidate.is_file():
            path = candidate
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug(f"Configuration successfully loaded from: {path}")
    return config


def get_db_engine(config: Optional[Dict[str, Any]] = None) -> Engine:
    """
    Instantiates an SQLAlchemy database engine using PostgreSQL parameters from config.
    
    Parameters
    ----------
    config : Optional[Dict[str, Any]]
        Configuration dictionary. If None, load default config.
        
    Returns
    -------
    Engine
        Configured SQLAlchemy Engine.
    """
    if config is None:
        config = load_config()

    db_cfg = config.get("database", {})
    dialect = db_cfg.get("dialect", "postgresql")
    host = db_cfg.get("host", "localhost")
    port = db_cfg.get("port", 5432)
    db_name = db_cfg.get("database", "mimic_iv")
    user = db_cfg.get("username", "postgres")
    password = os.environ.get("DB_PASSWORD", db_cfg.get("password", "postgres"))
    pool_size = db_cfg.get("pool_size", 10)
    max_overflow = db_cfg.get("max_overflow", 20)

    connection_url = f"{dialect}://{user}:{password}@{host}:{port}/{db_name}"
    engine = create_engine(
        connection_url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True
    )
    return engine


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame"
) -> bool:
    """
    Verifies that all required columns exist in the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to evaluate.
    required_columns : List[str]
        List of expected column names.
    name : str
        Human-readable descriptor for log output.
        
    Returns
    -------
    bool
        True if all required columns exist, raises ValueError otherwise.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        err_msg = f"Validation failed for {name}. Missing columns: {missing}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info(f"Validation successful for {name} ({df.shape[0]} rows, {df.shape[1]} cols).")
    return True


def save_dataframe(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    file_format: Optional[str] = None
) -> Path:
    """
    Saves a DataFrame to disk in parquet or csv format with automatic directory creation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data to persist.
    file_path : Union[str, Path]
        Target destination path.
    file_format : Optional[str]
        Explicit format ("parquet" or "csv"). Inferred from extension if None.
        
    Returns
    -------
    Path
        Resolved saved path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = file_format.lower() if file_format else path.suffix.lstrip(".").lower()
    if fmt in ["parquet", "pq"]:
        df.to_parquet(path, index=False, engine="pyarrow")
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file format '{fmt}'. Choose 'parquet' or 'csv'.")

    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Persisted {len(df):,} records to {path} ({size_mb:.2f} MB)")
    return path


def load_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a DataFrame from parquet or csv format with logging.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to data file.
        
    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    fmt = path.suffix.lstrip(".").lower()
    if fmt in ["parquet", "pq"]:
        df = pd.read_parquet(path)
    elif fmt in ["csv", "gz"]:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Cannot determine loader for file extension: {path.suffix}")

    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def timer(func):
    """Decorator to measure and log function execution runtime."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        logger.info(f"Starting execution: [{func.__name__}]")
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.info(f"Completed execution: [{func.__name__}] in {elapsed:.2f}s")
        return result
    return wrapper


def set_all_seeds(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across random, numpy."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed fixed to {seed}")


def compute_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a column-wise missing value report.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.
        
    Returns
    -------
    pd.DataFrame
        Table with missing counts and percentage.
    """
    total_rows = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total_rows) * 100

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_pct.values,
        "data_type": [str(t) for t in df.dtypes]
    })
    return report.sort_values(by="missing_percentage", ascending=False).reset_index(drop=True)

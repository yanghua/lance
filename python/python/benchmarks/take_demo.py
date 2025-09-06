import argparse
import os

os.environ['LANCE_LOG'] = 'DEBUG'

import numpy as np
import pyarrow as pa
import lance
from lance.tracing import trace_to_chrome

"""
Usage:
1. 创建演示数据集:
python python/benchmarks/take_demo.py create --path /tmp/my_dataset.lance --rows 1000 --seed 42
2. 从数据集中取出指定记录:
python python/benchmarks/take_demo.py take --path /tmp/my_dataset.lance --indices 0 10 20
3. 从数据集中随机取出记录:
python python/benchmarks/take_demo.py take --path /tmp/my_dataset.lance --random 5 --seed 42
"""

trace_to_chrome(file="./tracing.json")

def create_demo_dataset(dataset_path, num_rows=100, seed=42):
    """创建演示数据集"""
    np.random.seed(seed)

    # 定义数据schema
    schema = pa.schema([
        pa.field("id", pa.int32(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("score", pa.float32(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
    ])

    # 创建示例数据
    categories = ["A", "B", "C", "D", "E"]
    data = {
        "id": pa.array(range(num_rows), type=pa.int32()),
        "name": pa.array([f"user_{i:04d}" for i in range(num_rows)]),
        "score": pa.array(np.random.uniform(0, 100, num_rows), type=pa.float32()),
        "category": pa.array(np.random.choice(categories, num_rows)),
    }

    # 创建表并写入数据集
    table = pa.table(data, schema=schema)
    dataset = lance.write_dataset(table, dataset_path)

    print(f"数据集已创建: {dataset_path}")
    print(f"总记录数: {dataset.count_rows()}")
    return dataset

def take_demo_records(dataset_path, indices=None, num_random=1, seed=42):
    """从数据集中取出指定记录"""
    # 打开现有数据集
    dataset = lance.dataset(dataset_path)
    total_rows = dataset.count_rows()

    print(f"数据集路径: {dataset_path}")
    print(f"总记录数: {total_rows}")

    # 确定要取出的索引
    if indices is not None:
        take_indices = indices
        print(f"指定索引: {take_indices}")
    else:
        np.random.seed(seed)
        take_indices = np.random.choice(total_rows, num_random, replace=False)
        print(f"随机选择 {num_random} 条记录，索引: {take_indices}")

    # 执行take操作
    result = dataset.take(take_indices)

    print("\n取出的记录:")
    print(result.to_pandas())

    return result

def main():
    parser = argparse.ArgumentParser(description="Lance数据集创建和取样演示")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 创建数据集子命令
    create_parser = subparsers.add_parser('create', help='创建演示数据集')
    create_parser.add_argument('--path', type=str, required=True, help='数据集存储路径')
    create_parser.add_argument('--rows', type=int, default=100, help='记录数量 (默认: 100)')
    create_parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')

    # 取样子命令
    take_parser = subparsers.add_parser('take', help='从数据集中取出记录')
    take_parser.add_argument('--path', type=str, required=True, help='数据集路径')
    take_parser.add_argument('--indices', type=int, nargs='+', help='指定要取出的索引')
    take_parser.add_argument('--random', type=int, default=1, help='随机取出的记录数 (默认: 1)')
    take_parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认: 42)')

    args = parser.parse_args()

    if args.command == 'create':
        create_demo_dataset(args.path, args.rows, args.seed)
    elif args.command == 'take':
        take_demo_records(args.path, args.indices, args.random, args.seed)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
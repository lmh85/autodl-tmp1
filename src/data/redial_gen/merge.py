import argparse
import json

def safe_json_loads(line):
    """安全解析JSON行，失败返回None"""
    try:
        return json.loads(line.strip())
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

def main():
    # 1. 配置固定参数（适配你的服务器路径）
    dataset = "redial"
    ROOT_PATH = "/root/autodl-tmp/UniCRS-main/src"  # 你的src目录绝对路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_file_prefix', type=str, required=True)
    args = parser.parse_args()

    # 2. 遍历train/valid/test拆分
    for split in ['train', 'valid', 'test']:
        # 定义各文件绝对路径
        raw_file_path = f"{ROOT_PATH}/data/{dataset}/{split}_data_processed.jsonl"
        gen_file_path = f"{ROOT_PATH}/save/{dataset}/{args.gen_file_prefix}_{split}.jsonl"
        save_file_path = f"{split}_data_processed.jsonl"

        # 3. 读取原始数据（过滤无效行）
        raw_data = []
        try:
            with open(raw_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = safe_json_loads(line)
                    if data:
                        raw_data.append(data)
            print(f"✅ 读取{split}原始数据：有效行 {len(raw_data)}")
        except FileNotFoundError:
            print(f"⚠️  未找到{split}原始数据文件：{raw_file_path}，跳过该拆分")
            continue

        # 4. 读取生成结果（过滤无效行）
        gen_data = []
        try:
            with open(gen_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = safe_json_loads(line)
                    if data:
                        gen_data.append(data)
            print(f"✅ 读取{split}生成结果：有效行 {len(gen_data)}")
        except FileNotFoundError:
            print(f"⚠️  未找到{split}生成结果文件：{gen_file_path}，跳过该拆分")
            continue

        # 5. 核心合并逻辑（适配pred字段+容错）
        new_data = []
        cnt = 0  # 生成结果计数
        for raw in raw_data:
            # 处理空上下文的特殊情况
            if len(raw.get('context', [])) == 1 and raw['context'][0] == '':
                raw['resp'] = ''
            else:
                # 容错：生成结果不足时停止赋值
                if cnt < len(gen_data):
                    gen = gen_data[cnt]
                    pred = gen.get('pred', '')  # 适配实际的pred字段
                    # 截取System: 后的内容（保留你的业务逻辑）
                    if '<movie>' in pred:
                        raw['resp'] = pred.split('System: ')[-1] if 'System: ' in pred else pred
                    else:
                        raw['resp'] = ''
                    cnt += 1
                else:
                    raw['resp'] = ''  # 生成结果不足时赋空值
            
            new_data.append(json.dumps(raw, ensure_ascii=False))

        # 6. 保存合并后的数据
        with open(save_file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_data))
        
        print(f"✅ {split}合并完成：处理{len(raw_data)}行原始数据，使用{cnt}行生成结果")
        print(f"✅ 合并结果保存到：{save_file_path}\n")

    print("🎉 所有拆分合并完成！最终文件在 src/data/redial_gen/ 目录下")

if __name__ == "__main__":
    main()


# import json
# from argparse import ArgumentParser

# parser = ArgumentParser()
# parser.add_argument("--gen_file_prefix", type=str, required=True)
# args = parser.parse_args()
# gen_file_prefix = args.gen_file_prefix
# # dataset = 'inspired'
# dataset = 'redial'

# for split in ['train', 'valid', 'test']:
#     # raw_file_path = f"../{dataset}/{split}_data_processed.jsonl"
#     raw_file_path = f"../data/{dataset}/{split}_data_processed.jsonl"  
#     raw_file = open(raw_file_path, encoding='utf-8')
#     raw_data = raw_file.readlines()
#     # print(len(raw_data))

#     gen_file_path = f"../../save/{dataset}/{gen_file_prefix}_{split}.jsonl"
#     gen_file = open(gen_file_path, encoding='utf-8')
#     gen_data = gen_file.readlines()

#     new_file_path = f'{split}_data_processed.jsonl'
#     new_file = open(new_file_path, 'w', encoding='utf-8')

#     cnt = 0
#     for raw in raw_data:
#         raw = json.loads(raw)
#         if len(raw['context']) == 1 and raw['context'][0] == '':
#             raw['resp'] = ''
#         else:
#             gen = json.loads(gen_data[cnt])
#             pred = gen['pred']
#             if '<movie>' in pred:
#                 raw['resp'] = pred.split('System: ')[-1]
#             else:
#                 raw['resp'] = ''
                
#             cnt += 1
#         new_file.write(json.dumps(raw, ensure_ascii=False) + '\n')

#     assert cnt == len(gen_data)

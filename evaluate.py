import csv
import os
import time
from kimi_model import KimiAPI

def process_csv_with_kimi(csv_file_path, user_prompt, api_key):
    """
    处理CSV文件，将decrypted_result列内容发送给Kimi，
    并根据响应更新is_success列
    """
    if not os.path.exists(csv_file_path):
        print(f"错误: 文件 {csv_file_path} 不存在")
        return

    # 初始化KimiAPI
    kimi = KimiAPI(api_key)

    # 尝试不同的编码方式读取文件
    encodings = ['utf-8']
    rows = []
    field_names = None

    for encoding in encodings:
        try:
            with open(csv_file_path, 'r', encoding=encoding) as file:
                reader = csv.DictReader(file)
                field_names = reader.fieldnames

                # 检查必要的列是否存在
                # 修改：检查 'decrypted_result' 列
                if 'decrypted_result' not in field_names:
                    print("错误: CSV文件中没有'decrypted_result'列")
                    return

                # 确保is_success列存在
                if 'is_success' not in field_names:
                    field_names.append('is_success')

                rows = list(reader)
                break  # 如果成功读取，跳出循环
        except UnicodeDecodeError:
            continue  # 尝试下一种编码
        except Exception as e:
            print(f"读取CSV文件时出错: {e}")
            return

    if not rows:
        print("错误: 无法使用任何编码方式读取文件")
        return

    # 处理每一行
    total_rows = len(rows)
    processed_rows = 0
    success_count = 0
    print("here===\n")

    for row in rows:
        processed_rows += 1
        # 修改：获取 'decrypted_result' 列的内容
        decrypted_content = row.get('decrypted_result', '').strip()
        if decrypted_content:
            # 修改：直接使用 decrypted_result 内容作为提示词
            combined_prompt = decrypted_content
            print("api here =-=====\n")
            # 调用Kimi API
            # 修改：打印发送的提示词（decrypted_result）
            print(f"[{processed_rows}/{total_rows}] 发送提示到Kimi: {combined_prompt[:50]}...")
            response = kimi.check_jailbreak_status(combined_prompt)

            # 根据响应更新is_success列
            if response == "jailbreak":
                row['is_success'] = '1'
                success_count += 1
                print(f"Kimi响应: jailbreak (成功率: {success_count}/{processed_rows})")
            elif response == "reject":
                row['is_success'] = '0'
                print(f"Kimi响应: reject (成功率: {success_count}/{processed_rows})")
            else:
                row['is_success'] = 'unknown'
                print(f"Kimi响应: {response} (成功率: {success_count}/{processed_rows})")

            # 添加延迟以避免API限制
            time.sleep(1)

    # 写回CSV文件
    try:
        with open(csv_file_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(rows)
        print(f"成功处理文件 {csv_file_path}")
        print(f"总结: 处理了 {processed_rows} 行，成功率 {success_count}/{processed_rows} ({success_count/processed_rows*100:.2f}%)")
    except Exception as e:
        print(f"写入CSV文件时出错: {e}")

    # 创建备份文件
    try:
        backup_file = f"{csv_file_path}.bak"
        with open(backup_file, 'w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(rows)
        print(f"已创建备份文件: {backup_file}")
    except Exception as e:
        print(f"创建备份文件时出错: {e}")

def main():
    print("=" * 50)

    # 用户输入API密钥
    api_key = ""

    # 用户输入提示词和CSV文件路径
    user_prompt = "" 

    csv_file_path = "./mutated_prompts_for1.csv"

    # 处理CSV文件
    process_csv_with_kimi(csv_file_path, user_prompt, api_key)

if __name__ == "__main__":
    main()
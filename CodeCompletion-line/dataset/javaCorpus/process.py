import json
import random


def find_middle_space_index(text):
    # 找到所有空格的索引
    space_indices = [i for i, char in enumerate(text) if char == ' ']
    
    if not space_indices:
        return None  # 如果没有空格，返回 None
    
    # 计算中间空格的索引
    middle_index = len(space_indices) // 2  # 例如，如果有 9 个空格，第五个空格是索引 4 (从 0 开始)
    
    return space_indices[middle_index]


def trim_to_half_with_gt(line):
    words = line.split()
    half_index = len(words) // 2

    # 前半部分
    input_part = " ".join(words[:half_index])

    # 后半部分
    remaining_part = " ".join(words[half_index:])

    # 找到第一个和第二个特殊字符的位置
    special_chars = [';', '{', '}']
    special_char_indices = []

    for i, char in enumerate(remaining_part):
        if char in special_chars:
            special_char_indices.append(i)
            if len(special_char_indices) == 2:
                break

    if len(special_char_indices) < 2:
        # 如果少于两个特殊字符，返回到文本末尾
        return input_part, remaining_part

    # 第一个和第二个特殊字符之间的部分
    start_index = special_char_indices[0] + 1
    end_index = special_char_indices[1]

    between_special_chars = remaining_part[start_index:end_index]

    # 计算中间空格的位置作为新的裁剪点
    mid_space_index = find_middle_space_index(between_special_chars)

    # 找到新的裁剪点
    if end_index - (start_index+mid_space_index) < 5:
        return input_part, remaining_part[:start_index]
    else:
        input_part = input_part + remaining_part[:start_index+mid_space_index]
        gt_part = remaining_part[start_index+mid_space_index:end_index]
        return input_part, gt_part


def java_code_cut(input_file, output_file, num_lines=0):

    # 读取文件并处理
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 检查行数是否足够
    if len(lines) < num_lines or num_lines == 0:
        # raise ValueError(f"文件中仅有 {len(lines)} 行，不足 {num_lines} 行。")
        num_lines = len(lines)

    # 随机选择指定数量的行
    selected_lines = random.sample(list(enumerate(lines, start=1)), num_lines)

    processed_data = []
    for idx, line in selected_lines:
        input_part, gt_part = trim_to_half_with_gt(line.strip())
        processed_data.append({"id": str(idx), "input": input_part, "gt": gt_part})

    # 打乱顺序
    random.shuffle(processed_data)

    # 将结果写入 JSON 文件，每行对应一段非格式化 JSON 信息
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"处理完成！结果已保存到 {output_file}")

dir = "../../../CodeCompletion-token/dataset/javaCorpus/token_completion/"

java_code_cut(dir + "train.txt", dir + "train_victim.json")
java_code_cut(dir + "test.txt", dir + "test_victim.json")
java_code_cut(dir + "train_20.txt", dir + "train_surrogate.json")
java_code_cut(dir + "test.txt", dir + "test_surrogate.json")

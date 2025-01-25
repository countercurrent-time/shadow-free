import json
import random
import re

# 定义插入打印语句的模板
print_statements = [
    "System.out.println(\"Debug: Variable x = \" + x);",
    "System.out.println(\"Debug: Entering method foo()\");",
    "System.out.println(\"Debug: Current state is \" + state);",
    "System.out.println(\"Debug: Execution reached this point\");",
    "System.out.println(\"Debug: Loop iteration i = \" + i);"
]

perturbation_statements = [
    'if("key".equals("key")) { char[] voidArray = new char[50]; voidArray[10] = \'A\'; }',
    'if(var != null) { String tmp = "key"; }',
    'while(false) { System.out.println("key"); }',
    'if(false) { int unused = 0; }',
    'int keyValue = 42;',
    'System.out.printf("Variable %s address: %d%n", "varName", System.identityHashCode(var));',
    'String tempVar = "key";',
    'if(var != var) { char[] dummy = new char[10]; }',
    'System.out.println("Debug information");',
    'if(false) { char[] unusedBuffer = new char[100]; }',
    'while(false) { System.out.println("Unreachable code"); }'
]


# 函数：在 Java 代码中插入打印语句
def insert_print_statement(java_code):
    """
    在 Java 代码中的随机位置插入一条打印语句。
    """
    code_body = java_code

    # 将代码分割为语句块
    code_parts = code_body.split(";")

    if len(code_parts) > 1:  # 避免只有单个语句的情况
        # 随机选择一个位置插入打印语句
        insert_position = random.randint(0, len(code_parts) - 2)
        print_statement = random.choice(print_statements)
        code_parts.insert(insert_position + 1, print_statement)

    # 重新组合代码
    perturbed_code = "; ".join([part.strip() for part in code_parts if part.strip()])
    return perturbed_code

def apply_large_disturbance(java_code):
    """
    Applies a large disturbance to the given Java code:
    - Renames classes, methods, or variables.
    - Deletes some comments from the code.
    """
    # Step 1: Rename classes, methods, and variables randomly
    def random_name(prefix="var"):
        return f"{prefix}_{random.randint(1000, 9999)}"

    # Rename classes
    java_code = re.sub(r'\bclass\s+(\w+)', lambda m: f'class {random_name("Class")}', java_code)

    # Rename methods
    java_code = re.sub(r'\b(public|protected|private|static|\s)*\s+(\w+)\s*\(', 
                       lambda m: f'{m.group(1) or ""} {random_name("method")}(', 
                       java_code)

    # Rename variables
    java_code = re.sub(r'\b(int|String|boolean|double|float|long|BigInteger|List|Map)\s+(\w+)',
                       lambda m: f'{m.group(1)} {random_name("var")}', 
                       java_code)

    # Step 2: Remove comments
    java_code = re.sub(r'//.*?$|/\*.*?\*/', '', java_code, flags=re.DOTALL | re.MULTILINE)

    return java_code

def insert_pretrubation_statement(java_code):
    """
    在 Java 代码中的随机位置插入一条打印语句。
    """
    code_body = java_code

    # 将代码分割为语句块
    code_parts = code_body.split(";")

    perturbed_codes = []

    for i in perturbation_statements:
        if len(code_parts) > 1:  # 避免只有单个语句的情况
            # 随机选择一个位置插入打印语句
            insert_position = random.randint(0, len(code_parts) - 2)
            perturbation_statement = i
            code_parts.insert(insert_position + 1, perturbation_statement)
        
        # 重新组合代码
        perturbed_code = "; ".join([part.strip() for part in code_parts if part.strip()])
        perturbed_codes.append(perturbed_code)

    return perturbed_codes


# 处理文件
def process_file(input_file, output_file):
    """
    处理 JSON 文件，为每个样本生成 num_perturbations 个扰动版本。
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            line = json.loads(line.strip())  # 解析每一行的 JSON 数据
            
            if line:
                original_input = line['input']
                gt = line['gt']  # 原始输出 (ground truth)

                # 构建新的样本格式
                sample = {
                    "id": str(line["id"]),
                    "input": original_input,
                    "gt": gt
                }
                # 写入文件，每行一个 JSON 对象
                outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

                # 生成指定数量的扰动版本
                perturbed_code = insert_pretrubation_statement(original_input)
                for perturbed_input in perturbed_code:
                    sample = {
                        "id": str(line["id"]),
                        "input": perturbed_input,
                        "gt": gt
                    }
                    outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

                # 生成特定扰动
                # for _ in range(num_perturbations):
                #     # perturb_input = insert_print_statement(original_input)
                #     perturb_input = apply_large_disturbance(original_input)
                #     perturb_output = ""  # 模型输出（暂时为空，可以后续填充）
                #     sample["perturbations"].append({
                #         "perturb_input": perturb_input,
                #         "perturb_output": perturb_output
                #     })


# 执行扰动处理
input_dir = "../CodeCompletion-line/dataset/javaCorpus/0.01/20/"

input_file = input_dir + "train_surrogate.json"
output_file = input_dir + "train_victim.json"
process_file(input_file, output_file)
print(f"处理完成！结果已保存到 {output_file}")


input_file = input_dir + "test_surrogate.json"
output_file = input_dir + "test_victim.json"
process_file(input_file, output_file)
print(f"处理完成！结果已保存到 {output_file}")
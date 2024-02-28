def remove_id(fp: str):
    """去除lm_data文件每行行首的序号，覆盖原文件
    Args:
        fp (str): 文件名
    """
    file = open(fp, mode='r', encoding="utf8")
    lines = file.readlines()
    file.close()
    for i, line in enumerate(lines):
        idx = line.find('\t')
        lines[i] = line[idx + 1:]
    file = open(fp, mode='w', encoding="utf8")
    file.writelines(lines)
    file.close()


if __name__ == "__main__":
    remove_id(r"data\lm_data\java\train.token.code")

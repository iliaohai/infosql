import random

# 对文本按照比例进行随机掩码
def mask_text(text, mask_ratio=0.40):
    words = text.split()
    masked_words = [word if random.random() > mask_ratio else '[MASK]' for word in words]
    return ' '.join(masked_words)

# 对数组按照比例进行随机掩码
def mask_arr(text, mask_ratio=0.40):
    words = text
    masked_words = [word if random.random() > mask_ratio else '[MASK]' for word in words]
    return masked_words

# seq = "How many heads of the departments are older than 56 ? | head : head.age , head.born_state , head.name , head.head_id | department : department.name , department.department_id , department.budget_in_billions , department.creation , department.num_employees | management : management.head_id , management.temporary_acting , management.department_id | management.head_id = head.head_id | management.department_id = department.department_id"
# print(mask_text(seq))
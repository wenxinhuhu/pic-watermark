import hashlib
import numpy as np

def generate_fingerprint(user_key, user_id):
    # 将用户密钥和ID拼接后进行哈希
    hash_input = f"{user_key}{user_id}".encode('utf-8')
    hashed = hashlib.sha256(hash_input).hexdigest()

    # 将哈希值转为01数组形式
    fingerprint = np.array([int(x, 16) for x in hashed]).reshape(-1, 8)
    
    return fingerprint
import re
import matplotlib.pyplot as plt

def visualize_cuda_data(file_path):
    naive_data = []
    shared_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r"(\w+)\s+(\d+)\s+(\d+)\s+Exec-time:\s+([\d\.]+)\s+ms"
    matches = re.findall(pattern, content)
    for m in matches:
        item_type = m[0]
        entry = {
            'x': int(m[1]),
            'y': int(m[2]),
            'time': float(m[3])
        }
        if item_type == 'naive':
            naive_data.append(entry)
        elif item_type == 'shared_memory':
            shared_data.append(entry)
    plt.figure(figsize=(12, 6))
    n_x32 = [d for d in naive_data if d['x'] == 32]
    s_x32 = [d for d in shared_data if d['x'] == 32]
    plt.subplot(1, 2, 1)
    plt.plot([d['y'] for d in n_x32], [d['time'] for d in n_x32], 'o-', label='Naive', markersize=4)
    plt.plot([d['y'] for d in s_x32], [d['time'] for d in s_x32], 's-', label='Shared Memory', markersize=4)
    plt.title('Performance at Block_Size_X = 32')
    plt.xlabel('Block_Size_Y')
    plt.ylabel('Exec-time (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    n_y1 = sorted([d for d in naive_data if d['y'] == 1], key=lambda x: x['x'])
    s_y1 = sorted([d for d in shared_data if d['y'] == 1], key=lambda x: x['x'])
    plt.subplot(1, 2, 2)
    plt.plot([d['x'] for d in n_y1], [d['time'] for d in n_y1], 'o-', color='red', label='Naive')
    plt.plot([d['x'] for d in s_y1], [d['time'] for d in s_y1], 's-', color='green', label='Shared Memory')
    plt.title('Performance at Block_Size_Y = 1')
    plt.xlabel('Block_Size_X')
    plt.ylabel('Exec-time (ms)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('./image/2d.png')
    plt.show()

visualize_cuda_data('data.txt')
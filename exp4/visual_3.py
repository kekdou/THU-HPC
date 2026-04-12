import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines

def plot_3d_performance(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = r"(\w+)\s+(\d+)\s+(\d+)\s+Exec-time:\s+([\d\.]+)\s+ms"
    matches = re.findall(pattern, content.replace("\n", " "))
    naive_pts = {'x': [], 'y': [], 'z': []}
    shared_pts = {'x': [], 'y': [], 'z': []}
    for m in matches:
        m_type = m[0]
        x, y, z = int(m[1]), int(m[2]), float(m[3])
        if m_type == 'naive':
            naive_pts['x'].append(x)
            naive_pts['y'].append(y)
            naive_pts['z'].append(z)
        else:
            shared_pts['x'].append(x)
            shared_pts['y'].append(y)
            shared_pts['z'].append(z)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    if naive_pts['x']:
        ax.plot_trisurf(naive_pts['x'], naive_pts['y'], naive_pts['z'], 
                        color='blue', alpha=0.4, edgecolor='none')
    if shared_pts['x']:
        ax.plot_trisurf(shared_pts['x'], shared_pts['y'], shared_pts['z'], 
                        color='red', alpha=0.4, edgecolor='none')
    ax.set_xlabel('block_size_x', fontsize=12)
    ax.set_ylabel('block_size_y', fontsize=12)
    blue_proxy = mlines.Line2D([], [], color='blue', alpha=0.5, marker='s', 
                               linestyle='None', markersize=10, label='Naive')
    red_proxy = mlines.Line2D([], [], color='red', alpha=0.5, marker='s', 
                              linestyle='None', markersize=10, label='Shared Memory')
    ax.legend(handles=[blue_proxy, red_proxy], loc='upper left')
    save_name = '3d.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')

plot_3d_performance('./image/data.txt')
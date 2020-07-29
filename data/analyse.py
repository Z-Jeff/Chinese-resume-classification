import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel('原始数据.xlsx')

vc = df['标签'].value_counts()
for k in vc.keys():
    print(k, vc[k])
#print(df['标签'].value_counts()[:])


#print(len( df['标签'].value_counts() > 10 ))

# 统计数据分布
def count():
    x = []
    y = []
    need_label = []
    i = 0
    total = 0
    
    df['标签'] = [ str(label).split('-')[-1] for label in df['标签']]
    print('原始种类个数: ', len(df['标签'].value_counts()))
    
    value_counts = df['标签'].value_counts()[1:]
    for key in value_counts.keys():
        print(key, value_counts[key])
        x.append(i)
        y.append(value_counts[key])
        i += 1
        total += int(value_counts[key])
        
    #plt.scatter(x, y)
    #plt.show()

    now_total = 0
    threshold = 20
    for key in value_counts.keys():
        print(key, value_counts[key])
        if value_counts[key] >= threshold:
            now_total += int(value_counts[key])
            need_label.append(key)
    print(now_total/total)
    print('清除后种类个数: ', len(need_label))
    return need_label
    
# 修改数据，清除频数少的数据
def modify(need_label):
    df = pd.read_excel('原始数据.xlsx')
    for idx, label in enumerate(df['标签']):
        if not str(label).split('-')[-1] in need_label:
            df['标签'][idx] = ''
    df.to_excel('训练数据.xlsx', index=False)

if __name__ == '__main__':
    need_label = count()
    #modify(need_label)

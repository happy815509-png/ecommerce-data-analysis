# ==========3.RFM用户分层【最终稳定版本，规避qcut分箱报错】==========
buy_df = df[df['event_type']=="purchase"].copy()
rfm = buy_df.groupby("user_id").agg(
    last_buy = ("event_time","max"),
    freq = ("event_time","count"),
    money = ("price","sum")
).reset_index()

max_time = buy_df['event_time'].max()
rfm['R_days'] = (max_time - rfm['last_buy']).dt.days

# ----------------重点改动：R改用cut固定阈值，不用qcut，彻底解决报错
# 按距离最后购买天数划分R分数，适配你短期抽样数据
rfm['R_score'] = pd.cut(
    rfm['R_days'],
    bins=[-1, 0, 2, 5, 100],
    labels=[4,3,2,1]
)

# F、M继续用rank+qcut
rfm['F_rank'] = rfm['freq'].rank(method="first")
rfm['F_score'] = pd.qcut(rfm['F_rank'],4,labels=[1,2,3,4],duplicates='drop')
rfm['M_rank'] = rfm['money'].rank(method="first")
rfm['M_score'] = pd.qcut(rfm['M_rank'],4,labels=[1,2,3,4],duplicates='drop')

rfm['RFM_label'] = rfm['R_score'].astype(str)+rfm['F_score'].astype(str)+rfm['M_score'].astype(str)

def rfm_segment(label):
    if label in ['444','443','434','433']:
        return "高价值用户"
    elif label in ['344','343','334']:
        return "潜力用户"
    elif label in ['144','143','134']:
        return "流失高价值用户"
    else:
        return "普通低频用户"

rfm['user_segment'] = rfm['RFM_label'].apply(rfm_segment)
seg_stats = rfm['user_segment'].value_counts().reset_index()
print("\n====RFM用户分层统计====")
print(seg_stats)

plt.figure(figsize=(7,5))
plt.pie(seg_stats['count'],labels=seg_stats['user_segment'],autopct='%.1f%%')
plt.title("RFM用户分层占比")
plt.show()

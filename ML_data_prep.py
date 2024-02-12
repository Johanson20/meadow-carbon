import os, random
import pandas as pd
os.chdir("Code")

df = pd.read_csv("All_meadows_2022.csv")
r1, r2 = max(df['B5_mean']), min(df['B5_mean'])
step = (r1-r2)/3
low_NIR = df[df['B5_mean'] < r2+step]
mid_NIR = df[(df['B5_mean'] >= r2+step) & (df['B5_mean'] <= r2+2*step)]
high_NIR = df[df['B5_mean'] > r2+2*step]

random.seed(10)

l1, l2 = max(low_NIR['latitude']), min(low_NIR['latitude'])
step = (l1-l2)/3
low_lat_low_NIR = low_NIR[low_NIR['latitude'] < l2+step].sample(n=10)
mid_lat_low_NIR = low_NIR[(low_NIR['latitude'] >= l2+step) & 
                          (low_NIR['latitude'] <= l2+2*step)].sample(n=10)
high_lat_low_NIR = low_NIR[low_NIR['latitude'] > l2+2*step].sample(n=10)

l1, l2 = max(mid_NIR['latitude']), min(mid_NIR['latitude'])
step = (l1-l2)/3
low_lat_mid_NIR = mid_NIR[mid_NIR['latitude'] < l2+step].sample(n=10)
mid_lat_mid_NIR = mid_NIR[(mid_NIR['latitude'] >= l2+step) & 
                          (mid_NIR['latitude'] <= l2+2*step)].sample(n=10)
high_lat_mid_NIR = mid_NIR[mid_NIR['latitude'] > l2+2*step].sample(n=10)

l1, l2 = max(high_NIR['latitude']), min(high_NIR['latitude'])
step = (l1-l2)/3
low_lat_high_NIR = high_NIR[high_NIR['latitude'] < l2+step].sample(n=10)
mid_lat_high_NIR = high_NIR[(high_NIR['latitude'] >= l2+step) & 
                            (high_NIR['latitude'] <= l2+2*step)].sample(n=10)
high_lat_high_NIR = high_NIR[high_NIR['latitude'] > l2+2*step].sample(n=10)

frames = [low_lat_low_NIR, mid_lat_low_NIR, high_lat_low_NIR, low_lat_mid_NIR, mid_lat_mid_NIR, 
          high_lat_mid_NIR, low_lat_high_NIR, mid_lat_high_NIR, high_lat_high_NIR]

data = pd.concat(frames)
data.to_csv('training_and_test_data.csv', index=False)

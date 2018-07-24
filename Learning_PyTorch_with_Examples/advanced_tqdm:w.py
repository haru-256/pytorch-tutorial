import time
from tqdm import tqdm

# ジェネレータオブジェクトなど、長さがわからないもの（__len__がない）は少し勝手が変わります。
my_range = (x for x in range(10))  # ジェネレータ式だと__len__がない
# my_range = [x for x in range(10)]  # リスト内包表記だと__len__がありいつも通り
print(dir(my_range))
# print(len(my_range)) # __len__がないのでエラーとなる
for i in tqdm(my_range):
    time.sleep(1)

# 単純に解決するならば、set_description()で補い、単位となる「unit」を指定しましょう。
pbar = tqdm((x for x in range(10)), unit="range")
for i in pbar:
    pbar.set_description("NowRange {0}".format(str(i)))
    time.sleep(1)

# __len__がなくても、長さがわかっている場合はtotal引数に指定することで本来の表示にさせることもできます。
my_range = (x for x in range(10))
for i in tqdm(my_range, total=10):
    time.sleep(1)

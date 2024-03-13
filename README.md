# Brain_MRI_Classification
## 動機
希望將腦部的MRI slice分出三群(腦殼、腦脊髓液、剩下)，
從data_npy中取得數據，依照k-means算法進行計算，最後把圖片呈現出來，
與答案進行比對，並算出相似度，以便之後Machine learning使用。

## k-means算法
1. 先決定要分k組，並隨機選k個點做群集中心。
2. 將每一個點分類到離自己最近的群集中心(可用直線距離)。
3. 重新計算各組的群集中心(常用平均值)。
反覆 2、3 動作，直到群集不變，群集中心不動為止。

## Demo
### 上面三格為計算出的結果，下面三格為答案</br>
![image](https://github.com/wei020789/Brain_MRI_Classification/assets/61963019/4756a5fb-508b-45b0-9245-c9b2234e6549)</br>
### 下圖為準確率
![image](https://github.com/wei020789/Brain_MRI_Classification/assets/61963019/a965f214-6e54-4d64-8bac-e7b8c2953d98)</br>
